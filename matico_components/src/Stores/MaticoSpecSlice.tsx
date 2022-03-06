import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { Dashboard, Page } from "@maticoapp/matico_spec";
import _ from "lodash";
import { Dataset } from "../Datasets/Dataset";

export interface SpecState {
  spec: Dashboard | undefined;
  editing: boolean;
  currentEditPath?: string;
  currentEditType?: string;
}

const initialState: SpecState = {
  spec: undefined,
  editing: false,
};

const EditTypeMapping = {
  'pages': 'Page',
  'layers': 'Layer',
  'Map': 'Map',
  'sections': 'Section',
  'Text': 'Text',
  'Histogram':'Histogram',
  'PieChart':'PieChart',
  'Scatterplot':'Scatterplot',
  'Controls':'Controls'
}

const extractEditType = (path: string): string => {
  const parts = [...path.split(".")].reverse()
  const lastPart = parts.find(f => isNaN(+f))
  //@ts-ignore
  return EditTypeMapping[lastPart] || ''
}

const getParentPath = (path: string): string => {
  if (!path?.length) {
    return path
  }
  const parts = path.split(".")
  const reverseParts = [...parts].reverse()
  const lastPart = reverseParts.findIndex((f, i) => isNaN(+f) && !isNaN(+reverseParts[i - 1]))
  //@ts-ignore
  return parts.slice(0, parts.length-lastPart).join(".")
}

const incrementName = (name: string, takenNames: string[]): string => {
  //@ts-ignore
  const baseName = !isNaN(name.slice(-1)[0])
    ? name.slice(0, -1)
    : name;

  let tempName = `${baseName}`;
  let suffix = 2;
  do {
    tempName = `${baseName}${suffix}`;
    suffix++;
  } while (takenNames.includes(tempName));
  return tempName;
};

export const stateSlice = createSlice({
  name: "variables",
  initialState,
  reducers: {
    setEditing: (state, action: PayloadAction<boolean>) => {
      state.editing = action.payload;
    },
    setSpec: (state, action: PayloadAction<Dashboard>) => {
      state.spec = action.payload;
    },
    addDataset: (state, action: PayloadAction<{ dataset: Dataset }>) => {
      state.spec.datasets.push(action.payload.dataset);
    },
    addPage: (state, action: PayloadAction<{ page: Page }>) => {
      state.spec.pages.push(action.payload.page);
    },
    removePage: (state, action: PayloadAction<{ pageName: string }>) => {
      const newPages = state.spec.pages.filter(
        (p) => p.name !== action.payload.pageName
      );
      state.spec.pages = newPages;
    },
    setCurrentEditPath: (
      state,
      action: PayloadAction<{
        editPath: string | null;
        editType: string | null;
      }>
    ) => {
      state.currentEditPath = action.payload.editPath;
      state.currentEditType = extractEditType(action.payload.editPath);
    },
    setSpecAtPath: (
      state,
      action: PayloadAction<{ editPath: string; update: any }>
    ) => {
      let newSpec = { ...state.spec };
      if (action.payload.editPath === "") {
        newSpec = { ...newSpec, ...action.payload.update };
      } else {
        newSpec = { ...state.spec };
        const oldEntry = _.get(state.spec, action.payload.editPath);
        _.set(newSpec, action.payload.editPath, {
          ...oldEntry,
          ...action.payload.update,
        });
      }
      state.spec = newSpec;
      console.log("New spec is ", newSpec);
    },
    deleteSpecAtPath: (state, action: PayloadAction<{ editPath: string }>) => {
      const newSpec = { ...state.spec };
      _.unset(newSpec, action.payload.editPath);
      state.spec = newSpec;
    },
    duplicateSpecAtPath: (
      state,
      action: PayloadAction<{ editPath: string }>
    ) => { 
      const newSpec = { ...state.spec };
      const parentPath = getParentPath(action.payload.editPath);
      const takenNames = _.get(state.spec, parentPath)?.map(f => f.name);
      const oldEntry = _.get(state.spec, action.payload.editPath);
      const newEntry = {
        ..._.cloneDeep(oldEntry),
        name: incrementName(oldEntry.name, takenNames)
      }
      const oldParentEntries = _.get(state.spec, parentPath);

      _.set(newSpec, parentPath, [
        ...oldParentEntries,
        newEntry
      ]);
      state.spec = newSpec;
    },
    reconcileSpecAtPath: (
      state,
      action: PayloadAction<{ editPath: string, update: any }>
    ) => {
      let newSpec = { ...state.spec };
      if (action.payload.editPath === "") {
        newSpec = { ...newSpec, ...action.payload.update };
      } else {
        newSpec = { ...state.spec };
        const oldEntry = _.get(state.spec, action.payload.editPath);
        _.set(newSpec, action.payload.editPath, 
          _.merge(oldEntry, action.payload.update)
        );
      }
      state.spec = newSpec;
    },
  },
});

export const {
  setEditing,
  setSpec,
  addPage,
  removePage,
  setCurrentEditPath,
  setSpecAtPath,
  deleteSpecAtPath,
  duplicateSpecAtPath,
  reconcileSpecAtPath,
  addDataset,
} = stateSlice.actions;

export const selectSpec = (state: SpecState) => state.spec;

export const specReducer = stateSlice.reducer;
