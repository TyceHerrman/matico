import {View} from 'matico_spec';
export type StringVar = {
    type: 'string',
    name: string,
    value: string
}

export type NumberVar ={
    type :'number',
    name : string,
    value: number
}

export type AnyVar={
  type:any,
  name:string,
  value: any
}

export type MapLocVar={
    type: 'mapLocVar',
    name: string,
    value: {
      lat: number,
      lng: number,
      bearing: number,
      pitch:number,
      zoom:number
    } 
}

interface SelectionRange {
  type: "SelectionRange",
  variable: string;
  min: number;
  max: number;
}
interface NoSelection{
  type:"NoSelection"
}

type Selection = SelectionRange | NoSelection;

export type MaticoStateVariable = StringVar | NumberVar | MapLocVar | AnyVar | Selection;