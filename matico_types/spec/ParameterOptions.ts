import type { ColumnOptions } from "./ColumnOptions";
import type { NumericCategoryOptions } from "./NumericCategoryOptions";
import type { NumericFloatOptions } from "./NumericFloatOptions";
import type { NumericIntOptions } from "./NumericIntOptions";
import type { TableOptions } from "./TableOptions";
import type { TextCategoryOptions } from "./TextCategoryOptions";
import type { TextOptions } from "./TextOptions";

export type ParameterOptions = { type: "numericFloat" } & NumericFloatOptions | { type: "numericInt" } & NumericIntOptions | { type: "numericCategory" } & NumericCategoryOptions | { type: "textCategory" } & TextCategoryOptions | { type: "column" } & ColumnOptions | { type: "table" } & TableOptions | { type: "text" } & TextOptions;