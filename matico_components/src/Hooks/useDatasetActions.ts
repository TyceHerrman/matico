import { useMaticoDispatch, useMaticoSelector } from "./redux";
import { updateDatasetSpec } from "Stores/MaticoSpecSlice";

export const useDatasetActions = (name: string) => {
    const datasets = useMaticoSelector((s) => s.datasets);
    const dispatch = useMaticoDispatch();

    const updateDataset = (name: string, spec: any) => {
        dispatch(updateDatasetSpec({ name, datasetSpec: spec }));
    };
    return { updateDataset };
};
