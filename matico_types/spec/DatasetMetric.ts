import type { CategoriesParams } from "./CategoriesParams";
import type { EqualIntervalParams } from "./EqualIntervalParams";
import type { JenksParams } from "./JenksParams";
import type { QuantileParams } from "./QuantileParams";

export type DatasetMetric = { type: "min" } | { type: "max" } | { type: "quantile" } & QuantileParams | { type: "jenks" } & JenksParams | { type: "equalInterval" } & EqualIntervalParams | { type: "categories" } & CategoriesParams | { type: "mean" } | { type: "median" } | { type: "summary" };