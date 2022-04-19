use crate::db::DbPool;
use crate::models::SyncImport;
use actix::prelude::*;

use std::time::Duration;

pub struct ImportScheduler {
    pub db: DbPool,
    pub ogr_string: String,
    pub interval: Duration,
}

pub struct RunImportsMsg;

impl Message for RunImportsMsg {
    type Result = ();
}

impl Actor for ImportScheduler {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Context<Self>) {
        tracing::info!("Actor is alive");

        ctx.run_interval(self.interval, move |_act, ctx| {
            let my_addr = ctx.address();
            tracing::info!("Sending run import message");
            my_addr.do_send(RunImportsMsg {});
        });
    }
}

impl Handler<RunImportsMsg> for ImportScheduler {
    type Result = ();
    fn handle(&mut self, _msg: RunImportsMsg, ctx: &mut Context<Self>) -> Self::Result {
        tracing::info!("Running handle");
        let db_pool = self.db.clone();
        let ogr_string = self.ogr_string.clone();
        let execution = Box::pin(async move {
            tracing::info!("Schduling task");
            let requests = SyncImport::pending(&db_pool).expect("SOMHOW FAILED TO GET SYNC TABLE");
            tracing::info!("Pending jobs {:#?}", requests);

            for request in requests {
                match request.process(&db_pool, ogr_string.clone()).await {
                    Ok(_) => tracing::info!("Processed request {:?}", request),
                    Err(e) => tracing::info!("failed to process request {:?} with error {:?}", request, e),
                }
            }
            tracing::info!("All Done")
        });

        ctx.spawn(execution.into_actor(self));
        // actix::Arbiter::spawn(execution);
    }
}
