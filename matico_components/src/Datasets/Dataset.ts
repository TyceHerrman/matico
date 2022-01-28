
export enum DatasetState{
  LOADING = "LOADING",
  ERROR = "ERROR",
  READY = "READY" 
}
export interface Column{
    name: string,
    type : string
}

export enum GeomType{
  Point = "Point",
  Polygon = "Polygon",
  Line = "Line",
  None = "None"
}

//TODO Explicitly type this at some point
export type Datum = any;


export interface RangeFilter{
  variable: string,
  min?: number,
  max?: number
}

export interface CategoryFilter{
    variable: string,
    is_one_of?: Array<string>,
    is_not_one_of?: Array<string>
}

export type Filter = RangeFilter | CategoryFilter
export interface DatasetSummary{
  name: string,
  geomType?: GeomType,
  columns?: Array<Column>,
  local?:boolean,
  state: DatasetState
}

export interface Dataset{
  name: string,
  idCol: string,
  columns: ()=>Promise<Column[]>,
  getData: (filters?: Array<Filter>, columns?: Array<string>) =>Promise<Datum[]>,
  getDataWithGeo: (filters?: Array<Filter>, columns?:Array<string>) =>Promise<Datum[]>,
  getFeature: (feature_id: string) => Promise<Datum | undefined>,
  local:()=>boolean,
  tiled:()=>boolean,
  isReady:()=> boolean,
  geometryType:()=> Promise<GeomType>,
  onStateChange?:(reportState :(state: DatasetState)=>void)=>void,
  getColumnMax: (columns: string) => Promise<number>,
  getColumnMin: (columns: string) => Promise<number>,
  getColumnSum: (columns: string) => Promise<number>,
  getCategoryCounts: (columns: string, filters?: Array<Filter>) => Promise<{[entry: string| number]: number}>,
  getEqualIntervalBins:(column: string, bins:number, filters?: Array<Filter>)=> Promise<Array<number>>,
  getQuantileBins:(column: string, bins:number, filters?:Array<Filter>)=>Promise<Array<number>>,
  getJenksBins:(column: string, bins:number, filters?: Array<Filter>)=>Promise<Array<number[]>>

  // metricForColumn?: (columnName:string, metric: DatasetMetric)=> any
}
