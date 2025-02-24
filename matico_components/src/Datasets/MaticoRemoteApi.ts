import { Dataset, DatasetState } from "./Dataset";
import { Filter } from "@maticoapp/matico_types/spec";
import { Column, HistogramResults } from "@maticoapp/matico_types/api";

import axios from "axios";

const encodeParams = (params: { [param: string]: any }) => {
    const urlParams = Object.keys(params)
        .map(
            (key: string) =>
                `${encodeURIComponent(key)}=${encodeURIComponent(params[key])}`
        )
        .join("&");
    return urlParams;
};

export class MaticoRemoteApi implements Dataset {
    name: string;
    api_id: string;
    token: string;
    serverUrl: string;
    params: { [param: string]: any };
    idCol: string;
    _axiosInstance: any;

    constructor(
        name: string,
        api_id: string,
        serverUrl: string,
        params: { [param: string]: any },
        token?: string
    ) {
        this.name = name;
        this.api_id = api_id;
        this.params = params;
        this.serverUrl = serverUrl;
        this.token = token;

        const headers: { [header: string]: string } = {
            "Content-Type": "application/json"
        };
        if (token) {
            headers["Authorization"] = `Bearer ${token}`;
        }

        this._axiosInstance = axios.create({
            baseURL: serverUrl + `/apis/${api_id}`,
            headers
        });
    }

    raster() {
        return false;
    }
    async _queryServer(path: string, params?: { [param: string]: any }) {
        const queryResults = await this._axiosInstance(path, {
            params: { ...params, ...this.params }
        });
        return queryResults.data;
    }

    async getCategories(
        columns: string,
        noCategories: number,
        filters?: Filter[]
    ) {
        throw Error("not yet implmented");
        return Promise.resolve([]);
    }

    async columns() {
        const serverColumns = await this._queryServer("/columns");
        const typeMappings: { [postgisType: string]: any } = {
            INT4: "number",
            VARCHAR: "string",
            INT8: "number",
            FLOAT: "number"
        };
        return serverColumns.map((sc: Column) => ({
            name: sc.name,
            type: typeMappings[sc.colType] ?? sc.colType
        }));
    }

    getData(filters?: Filter[], columns?: string[], limit?: number) {
        return this._queryServer(`/run?${encodeParams(this.params)}`);
    }

    getDataWithGeo(filters?: Filter[], columns?: string[]) {
        return this._queryServer(`/run?${encodeParams(this.params)}`);
    }

    getFeature(feature_id: string) {
        return this._queryServer(`feature/${feature_id}`);
    }
    local() {
        return false;
    }
    tiled() {
        return true;
    }
    mvtUrl() {
        return `${this.serverUrl}/tiler/api/${
            this.api_id
        }/{z}/{x}/{y}?${encodeParams(this.params)}`;
    }
    isReady() {
        return true;
    }
    async geometryType() {
        let columns = await this.columns();
        return columns.find((c: Column) => c.colType === "geometry")?.type;
    }

    async getColumnMax(column: string) {
        const statParams = {
            stat: JSON.stringify({
                BasicStatParams: { treat_null_as_zero: true }
            })
        };

        let stats = await this._queryServer(
            `/columns/${column}/stats`,
            statParams
        );
        return stats.max;
    }

    async getColumnMin(column: string) {
        const statParams = {
            stat: JSON.stringify({
                BasicStatParams: { treat_null_as_zero: true }
            })
        };

        let stats = await this._queryServer(
            `/columns/${column}/stats`,
            statParams
        );
        return stats.min;
    }

    async getColumnSum(column: string) {
        const statParams = {
            stat: JSON.stringify({
                BasicStatParams: { treat_null_as_zero: true }
            })
        };

        let stats = await this._queryServer(
            `/columns/${column}/stats`,
            statParams
        );
        return stats.sum;
    }

    async getCategoryCounts(column: string, filters?: Filter[]) {
        const statParams = {
            stat: JSON.stringify({ ValueCounts: { ignore_null: true } })
        };

        let categoryCounts = await this._queryServer(
            `/columns/${column}/stats`,
            statParams
        );
        return categoryCounts;
    }

    async getColumnHistogram(
        column: string,
        noBins: number,
        filters?: Filter[]
    ) {
        const statParams = {
            stat: JSON.stringify({
                Histogram: { no_bins: noBins, treat_null_as_zero: false }
            })
        };

        let result: HistogramResults = await this._queryServer(
            `/columns/${column}/stats`,
            statParams
        );

        return result;
    }

    async getEqualIntervalBins(
        column: string,
        bins: number,
        filters?: Filter[]
    ) {
        const statParams = {
            stat: JSON.stringify({
                Percentiles: { no_bins: bins, treat_null_as_zero: false }
            })
        };

        let result = await this._queryServer(
            `/columns/${column}/stats`,
            statParams
        );
        return result;
    }

    // TODO implement backend quantile bins
    async getQuantileBins(column: string, bins: number, filters?: Filter[]) {
        const statParams = {
            stat: JSON.stringify({
                Quantiles: { no_bins: bins, treat_null_as_zero: false }
            })
        };

        let result = await this._queryServer(
            `/columns/${column}/stats`,
            statParams
        );
        return result.Quantiles.map((q: any) => q.bin_end);
    }
    //TODO implement backend jenks bins
    getJenksBins: (
        column: string,
        bins: number,
        filters?: Filter[]
    ) => Promise<number[][]>;

    //TODO implement invalidation and updates
    onStateChange(reportState: (state: DatasetState) => void) {}
}
