import { Page, Pane, ContainerPane, PaneRef } from "@maticoapp/matico_types/spec";
import { useMaticoDispatch, useMaticoSelector } from "./redux";
import {
    updatePageDetails,
    removePage,
    addPage,
    setCurrentEditElement,
    addPaneRefToContainer,
    removePaneFromContainer
} from "../Stores/MaticoSpecSlice";
import { v4 as uuidv4 } from "uuid";

export const useApp = () => {
    const { metadata, pages, panes, theme } = useMaticoSelector(
        (selector) => selector.spec.spec
    );
    const dispatch = useMaticoDispatch();

    const addPageLocal = (page: Partial<Page>) => {
        dispatch(
            addPage({
                page: {
                    name: pages.length === 0 ? "Home" : `Page ${pages.length}`,
                    id: uuidv4(),
                    layout: { type: "free" },
                    panes: [],
                    icon: pages.length ===0 ? "faHome" : "faPage",
                    path: pages.length ===0 ? "/" : `/page_${pages.length}`,
                    ...page
                }
            })
        );
    };

    const setEditPage = (id: string) => {
        dispatch(
            setCurrentEditElement({
                type: "page",
                id
            })
        );
    };

    const updatePage = (pageId: string, update:Partial<Page>)=>{
      dispatch(
        updatePageDetails({pageId, update})
      ) 
    }

    const removePageLocal = (id: string) => {
        dispatch(removePage({ id, removeOrphanPanes: false }));
    };

    const reparentPane = (
        pane: Pane,
        parent: ContainerPane | Page,
        target: ContainerPane | Page
    ) => {
        const paneRef = parent.panes.find((parentPane: PaneRef) => parentPane.paneId === pane.id)
        console.log(parent)
        dispatch(
            removePaneFromContainer({
                containerId:parent.id, 
                paneRefId:paneRef.id
            }))
        dispatch(
            addPaneRefToContainer({
                containerId: target.id,
                paneRef
        }))
    }

    return {
        pages,
        panes,
        theme,
        removePage: removePageLocal,
        addPage: addPageLocal,
        updatePage,
        setEditPage,
        reparentPane
    };
};
