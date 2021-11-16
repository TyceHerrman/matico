import React from 'react';
import { Vega } from 'react-vega';
import * as vega from 'vega';
// import { useSelector, useDispatch } from 'react-redux';
import {useContext, useEffect, useState, useRef, useMemo} from 'react';
import { MaticoDataContext } from "../../../Contexts/MaticoDataContext/MaticoDataContext";
import { MaticoPaneInterface } from "../Pane";
import { Box } from "grommet";
import { useAutoVariable } from "../../../Hooks/useAutoVariable";
import {Filter} from '../../../Datasets/Dataset';
// import styles from './Widgets.module.scss';
// import useGetScatterData from '@webgeoda/hooks/useGetScatterData';
// import usePanMap from '@webgeoda/hooks/usePanMap';
// import Loader from '@components/layout/Loade r';

const renderVega = (
    chartRef,
    spec,
    scatterData,
    signalListeners,
    setView
) => <Vega 
    ref={chartRef}
    spec={spec} 
    data={scatterData} 
    signalListeners={signalListeners} 
    onNewView={(view) => setView(view)}
    />  

// const parseFilters = (filters) => {
//     let returnObj = {}
//     for (let i=0; i<filters.length; i++){
//         returnObj[`${filters[i].id.slice(-1,)[0]}max`] = filters[i].to
//         returnObj[`${filters[i].id.slice(-1,)[0]}min`] = filters[i].from
//     }
//     return [returnObj]
// }
const backgroundColor="#fff" //todo hoist to pane spec

interface MaticoScatterplotPaneInterface extends MaticoPaneInterface {
    dataset: {name: string, filters: Array<Filter>};
    x_column: string;
    y_column: string;
    dot_color?: string;
    // backgroundColor: string;
    dot_size?: number;
}
  


export const MaticoScatterplotPane: React.FC<MaticoScatterplotPaneInterface> = ({
    dataset={},
    x_column='',
    y_column='',
    dot_color='#ff0000',
    dot_size=1
    // config={},
    // options={},
    // id=null
}) => {

    
    const { state: dataState } = useContext(MaticoDataContext);
    const [view, setView] = useState({});
    const chartRef = useRef();

    const [ 
        xFilter,
        updateXFilter,
          //@ts-ignore
    ] = useAutoVariable(
          //@ts-ignore
        `${x_column}_range`, 
          //@ts-ignore
        "NoSelection", 
        {
            type:"NoSelection"
        }
    )

    const [ 
        yFilter,
        updateYFilter,
          //@ts-ignore
    ] = useAutoVariable(
        `${y_column}_range`, 
          //@ts-ignore
        "NoSelection", 
        {
            type:"NoSelection"
        }
    )

    const foundDataset = dataState.datasets.find((d) => {
        return d.name === dataset.name;
    });

    if(!foundDataset){
      return<div>Loading</div>
    }
    
    // @ts-ignore
    const chartData = foundDataset && foundDataset.isReady() ? foundDataset.getData(dataset.filters) : []; // todo: centralized data model 
    
    const spec = {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "width": 300,
        "height": 160,
        "padding": {
        "top":    10,
        "left":   40,
        "bottom": 30,
        "right":  10
        },
        "autosize": "none",
        "config": {
        "axis": {
            "domain": false,
            "tickSize": 3,
            "tickColor": "#888",
            "labelFont": "Monaco, Courier New"
        }
        },
    
        "signals": [
            { "name": "chartWidth", "value": 300 },
            { "name": "chartHeight", "value": 160 },
            {
                "name": "hover",
                "on": [
                    // {"events": "*:mousedown", "encode": "select"},
                    // {"events": "*:mouseup",   "encode": "release"}
                ]
            },
            {
                "name": "click",
                "on": [
                {"events": "*:click", "encode": "click"}
                ]
            },
            {
                "name": "xoffset",
                "update": "-(height + padding.bottom)"
            },
            {
                "name": "yoffset",
                "update": "-(width + padding.left)"
            },
            { "name": "xrange", "update": "[0, width]" },
            { "name": "yrange", "update": "[height, 0]" },
            { "name": "xcur", "value": null, "update": "slice(xdom)"},
            { "name": "ycur", "value": null, "update": "slice(ydom)"}, 
            {
                "name": "startDrag", "value": null,
                "on": [
                    {"events": "mouseup, touchend", "update": "0"},
                    {"events": "mousedown, touchstart", "update": "1"},
                ]
            }, 
            {
                "name": "startDragCoords", "value": null,
                "on": [
                    {"events": "mousedown, touchstart", "update": "[invert('xscale', x()),invert('yscale', y())]"},
                ]
            },   
            {
                "name": "dragBox", "value": null,
                on: [{
                        "events": {"signal": "startDragCoords"},
                        "force": true,
                        "update":"[[startDragCoords[0], startDragCoords[1]],[]]"
                    },
                    {   
                        "events": "mousemove, touchmove", "update": "startDrag ? [dragBox[0],[invert('xscale', x()),invert('yscale', y())]] : dragBox"
                    }]
            }, 
            {
                "name": "endDrag", "value": null,
                "on": [
                    {"events": "mouseup, touchend", "update": "dragBox"},
                ]
            },
            {
                "name": "anchor", "value": [0, 0],
                // "on": [
                // {
                //     "events": "wheel",
                //     "update": "[invert('xscale', x()), invert('yscale', y())]"
                // },
                // {
                //     "events": {"type": "touchstart", "filter": "event.touches.length===2"},
                //     "update": "[(xdom[0] + xdom[1]) / 2, (ydom[0] + ydom[1]) / 2]"
                // }
                // ]
            },
            {
                "name": "zoom", "value": 1,
                // "on": [
                // {
                //     "events": "wheel!",
                //     "force": true,
                //     "update": "pow(1.001, event.deltaY * pow(16, event.deltaMode))"
                // },
                // {
                //     "events": {"signal": "dist2"},
                //     "force": true,
                //     "update": "dist1 / dist2"
                // }
                // ]
            },
            {
                "name": "dist1", "value": 0,
                // "on": [
                // {
                //     "events": {"type": "touchstart", "filter": "event.touches.length===2"},
                //     "update": "pinchDistance(event)"
                // },
                // {
                //     "events": {"signal": "dist2"},
                //     "update": "dist2"
                // }
                // ]
            },
            {
                "name": "dist2", "value": 0,
                // "on": [{
                // "events": {"type": "touchmove", "consume": true, "filter": "event.touches.length===2"},
                // "update": "pinchDistance(event)"
                // }]
            },
            {
                "name": "xdom", "update": "slice(xext)",
                // "on": [
                //     {
                //         "events": {"signal": "zoom"},
                //         "update": "[anchor[0] + (xdom[0] - anchor[0]) * zoom, anchor[0] + (xdom[1] - anchor[0]) * zoom]"
                //     }
                // ]
            },
            {
                "name": "ydom", "update": "slice(yext)",
                // "on": [
                //     {
                //         "events": {"signal": "zoom"},
                //         "update": "[anchor[1] + (ydom[0] - anchor[1]) * zoom, anchor[1] + (ydom[1] - anchor[1]) * zoom]"
                //     }
                // ]
            },
            // {
            //     "name": "size",
            //     "update": "clamp(100 / span(xdom), 5, 1000)"
            // },
        ],
        
            "data": [
                {
                    "name": "table",
                    "transform": [
                        { "type": "filter", "expr": "datum.x != 0 && datum.y != 0"},
                        { "type": "extent", "field": x_column, "signal": "xext" },
                        { "type": "extent", "field": y_column, "signal": "yext" },
                    ]
                },
                // config.aggregate !== undefined? 
                //     {
                //         "source": "table",
                //         "name": "tableBinned",
                //         "transform": [
                //             { "type": "filter", "expr": "datum.x != 0 && datum.y != 0"},
                //             {
                //                 "type": "bin", "field": "x", "maxbins": 40,
                //                 "extent": {"signal": "xext"},
                //                 "as": ["x0", "x1"]
                //             },
                //             {
                //                 "type": "bin", "field": "y", "maxbins": 20,
                //                 "extent": {"signal": "yext"},
                //                 "as": ["y0", "y1"]
                //             },
                //             {
                //                 "type": "aggregate",
                //                 "groupby": ["x0","x1","y0","y1"]
                //             } 
                //         ]
                //     }
                //     :   {"name": "tablebinned"}
                // ,
                // {
                //     "name": "active",
                //     "transform": [
                //         { "type": "filter", "expr": "datum.x != 0 && datum.y != 0"},
                //     ]
                // },
                // config.aggregate !== undefined ? 
                //     {
                //         "source": "active",
                //         "name": "activeBinned",
                //         "transform": [
                //             { "type": "filter", "expr": "datum.x != 0 && datum.y != 0"},
                //             {
                //                 "type": "bin", "field": "x", "maxbins": 40,
                //                 "extent": {"signal": "xext"},
                //                 "as": ["x0", "x1"]
                //             },
                //             {
                //                 "type": "bin", "field": "y", "maxbins": 20,
                //                 "extent": {"signal": "yext"},
                //                 "as": ["y0", "y1"]
                //             },
                //             {
                //                 "type": "aggregate",
                //                 "groupby": ["x0","x1","y0","y1"]
                //             } 
                //         ]
                //     }
                //     :   {"name": "activeBinned"}
                // ,
                // {
                //     "name":"filterExtent"
                // },
                // options.regression ? {
                //     "name": "trend",
                //     "source": "table",
                //     "transform": [
                //         {
                //             "type": "regression",
                //             "extent": {"signal": "xdom"},
                //             "x": "x",
                //             "y": "y",
                //             "as": ["u", "v"]
                //         }
                //     ]
                // } : {"name": "trend"},
                // options.regression ? {
                //     "name": "trend-active",
                //     "source": "active",
                //     "transform": [
                //         {
                //             "type": "regression",
                //             "extent": {"signal": "xdom"},
                //             "x": "x",
                //             "y": "y",
                //             "as": ["u", "v"]
                //         }
                //     ]
                // } : {"name": "trend-active"},
            ],
        
            "scales": [
                {
                    "name": "xscale", "zero": false,
                    "domain": {"signal": "xdom"},
                    "range": {"signal": "xrange"}
                },
                {
                    "name": "yscale", "zero": false,
                    "domain": {"signal": "ydom"},
                    "range": {"signal": "yrange"}
                },
                // config.aggregate === 'scale' ? {
                //     "name": "size",
                //     "type": "linear",
                //     "zero": true,
                //     "domain": {"data": "tableBinned", "field": "count"},
                //     "range": [0,360]
                // }
                // : config.aggregate === 'heatmap' ? {
                //     "name": "color",
                //     "type": "linear",
                //     "range": {"scheme": "Viridis"},
                //     "domain": {"data": "tableBinned", "field": "count"},
                //     "zero": false, 
                //     "nice": true
                // } : { 'name': 'null'}
            ],
        
            "axes": [
                {
                    "scale": "xscale",
                    "orient": "top",
                    "offset": {"signal": "xoffset+10"},
                    "labelOverlap": false,
                    "grid":true,
                    "format": ".2s",
                    "title":x_column,
                    "tickCount":5,
                    "tickColor":"#fff",
                    "titleY":10
                },
                {
                    "scale": "yscale",
                    "orient": "right",
                    "offset": {"signal": "yoffset+10"},
                    "title":y_column,
                    "labelOverlap": false,
                    "grid":true,
                    "format": ".2s",
                    "tickCount":3,
                    "tickColor":"#fff",
                    "titleX":-10
                }
        ],
    
        "marks": [
            {
                "type": "rect",
                "encode": {
                    "enter": {
                        "fill": {"value": "#0099FF55"},
                        "stroke": {"value": "#0099FF"}
                    },
                    "update": {
                        "x": {"scale":"xscale", "signal": "startDrag ? dragBox[0][0] : 0"},
                        "x2": {"scale":"xscale", "signal": "startDrag ? dragBox[1][0] : 0"},
                        "y": {"scale":"yscale", "signal": "startDrag ? dragBox[0][1] : 0"},
                        "y2": {"scale":"yscale", "signal": "startDrag ? dragBox[1][1] : 0"}
                    }
                }
            },
            // {
            //     "type": "rect",
            //     "from": {"data":"filterExtent"},
            //     "encode": {
            //         "enter": {
            //             "fill": {"value": "#0099FF55"},
            //             "stroke": {"value": "#0099FF"}
            //         },
            //         "update": {
            //             "x": {"scale": "xscale", "field": "xmin"},
            //             "x2": {"scale": "xscale", "field": "xmax"},
            //             "y": {"scale": "yscale", "field": "ymin"},
            //             "y2": {"scale": "yscale", "field": "ymax"},
            //         }
            //     }
            // },
            // config.aggregate === 'scale' ? {
            //     "type": "symbol",
            //     "from": {"data": "tableBinned"},
            //     "encode": {
            //       "update": {
            //         "x": {"scale": "xscale", "signal": "(datum.x0 + datum.x1) / 2"},
            //         "y": {"scale": "yscale", "signal": "(datum.y0 + datum.y1) / 2"},
            //         "size": {"scale": "size", "field": "count"},
            //         "shape": {"value": "circle"},
            //         "fill": {"value": "#4682b4"}
            //       }
            //     }
            // } : config.aggregate === 'heatmap' ? 
            // {
            //   "type": "rect",
            //   "from": {"data": "tableBinned"},
            //   "encode": {
            //     "enter": {
            //         "x": {"scale": "xscale", "field": "x0"},
            //         "x2": {"scale": "xscale", "field": "x1"},
            //         "y": {"scale": "yscale", "field": "y0"},
            //         "y2": {"scale": "yscale", "field": "y1"},
            //         "fill": {"scale": "color", "field": "count"}
            //     }
            //   }
            // } : 
            {
                "type": "symbol",
                "from": {"data": "table"},
                "clip": true,
                "encode": {
                "enter": {
                    "fillOpacity": {"value": 0.6},
                    "fill": {"value": dot_color},
                    "size": {"value": dot_size}
                },
                "update": {
                    "x": {"scale": "xscale", "field": x_column}, // TODO update x to relevant field for spec
                    "y": {"scale": "yscale", "field": y_column}, // TODO update y to relevant field for spec

                },
                // "select":  { "size": {"signal": "size", "mult": 5} },
                // "release": { "size": {"signal": "size"} }
                }
            },
            // config.aggregate === 'scale' ? {
            //     "type": "symbol",
            //     "from": {"data": "activeBinned"},
            //     "encode": {
            //       "update": {
            //         "x": {"scale": "xscale", "signal": "(datum.x0 + datum.x1) / 2"},
            //         "y": {"scale": "yscale", "signal": "(datum.y0 + datum.y1) / 2"},
            //         "size": {"scale": "size", "field": "count"},
            //         "shape": {"value": "circle"},
            //         "fill": {"value": "red"}
            //       }
            //     }
            // } : config.aggregate === 'heatmap' ? 
            // {
            //   "type": "rect",
            //   "from": {"data": "activeBinned"},
            //   "encode": {
            //     "enter": {
            //         "x": {"scale": "xscale", "field": "x0"},
            //         "x2": {"scale": "xscale", "field": "x1"},
            //         "y": {"scale": "yscale", "field": "y0"},
            //         "y2": {"scale": "yscale", "field": "y1"},
            //         "fill": {"scale": "color", "field": "count"}
            //     }
            //   }
            // } : 
            // {
            //     "type": "symbol",
            //     "from": {"data": "active"},
            //     "clip": true,
            //     "encode": {
            //         "enter": {
            //             "fillOpacity": {"value": 0.6},
            //             "fill": {"value": "red"}
            //         },
            //         "update": {
            //             "x": {"scale": "xscale", "field": "x"},
            //             "y": {"scale": "yscale", "field": "y"},
            //             "size": {"value": 20}
            //         },
            //         "select":  { "size": {"signal": "size", "mult": 5} },
            //         "release": { "size": {"signal": "size"} }
            //     }
            // },
            // options.regression ? {
            //     "type": "line",
            //     "from": {"data": "trend"},
            //     "clip":true,
            //     "encode": {
            //         "update": {
            //             "x": {"scale": "xscale", "field": "u"},
            //             "y": {"scale": "yscale", "field": "v"},
            //             "stroke": {"value": "#222"}
            //         }
            //     }
            // } : {"type": "line"},
            // options.regression ? {
            //     "type": "line",
            //     "from": {"data": "trend-active"},
            //     "clip":true,
            //     "encode": {
            //         "update": {
            //             "x": {"scale": "xscale", "field": "u"},
            //             "y": {"scale": "yscale", "field": "v"},
            //             "stroke": {"value": "red"}
            //         }
            //     }
            // } : {"type": "line"},
        ]
    }

    function handleDragEnd(e, result){
        if (isNaN(result[1][0])||isNaN(result[1][1])) return

        updateXFilter({
            type:"SelectionRange",
            variable: x_column,
            min: Math.min(result[0][0], result[1][0]),
            max: Math.max(result[0][0], result[1][0])
        })
        updateYFilter({
            type:"SelectionRange",
            variable: y_column,
            min: Math.min(result[0][1], result[1][1]),
            max: Math.max(result[0][1], result[1][1])

        })
    }
    

    const signalListeners = { 
        endDrag: handleDragEnd,
        // click: handleClick,
        // tempDrag: (e, target) => console.log(e, target)
    };

    const vegaChart = useMemo(() => renderVega(
        chartRef,
        spec,
        {table: chartData}, // vega needs datasets as an object
        signalListeners,
        setView
    ), [chartData.length, x_column, y_column, dot_color, dot_size, foundDataset?.isReady()])
    
    
    //
    if (foundDataset === undefined || !foundDataset || !foundDataset.isReady()) {
        return <div>{foundDataset.name} not found!</div>
    }

    
    // const boxFilterGeoids = useSelector((state) => state.boxFilterGeoids);
    // const panToGeoid = usePanMap();
    // const mapFilters = useSelector((state) => state.mapFilters);
    // const currFilters = mapFilters.filter(f => f.source === config.id);
    // const {
    //     chartData
    // } = useGetScatterData({
    //     config: config,
    //     options: options,
    //     id: id
    // })

    // useEffect(() => {
    //     if (!(chartData.length)) return;
    //     function updateGraph() {
    //         const cs = vega
    //             .changeset()
    //             .remove(() => true)
    //             .insert(chartData.filter(t => boxFilterGeoids.includes(t.id)));
    //         view.change('active', cs).run();
    //     }
    //     Object.keys(view).length && updateGraph();
    // }, [view, boxFilterGeoids]);


    // useEffect(() => {
    //     // if (currFilters.length === 1) return;
    //     function updateExtent() {
    //         const cs = vega
    //             .changeset()
    //             .remove(() => true)
    //             .insert(parseFilters(currFilters));
    //         view.change('filterExtent', cs).run();
    //     }
    //     Object.keys(view).length && updateExtent();
    //     // view && updateGraph();
    // }, [view, currFilters]);
    
    
    return (
        <Box background={backgroundColor} elevation={"large"} fill={true} overflow={{vertical:'auto'}}>
            {chartData.length 
                ? vegaChart
                : null
            } 
        </Box>  
    );
}

// ScatterWidget.propTypes = {
//   options: PropTypes.object.isRequired,
//   data: PropTypes.object.isRequired,
//   id: PropTypes.number.isRequired,
//   config: PropTypes.object.isRequired,
//   activeFilters: PropTypes.array.isRequired
// };
