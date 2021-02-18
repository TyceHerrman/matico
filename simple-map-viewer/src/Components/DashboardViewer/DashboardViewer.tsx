import React, { useState } from 'react';
import ReactMapGL from 'react-map-gl';
import { Styles } from './DashboardViewerStyles';
import { Dashboard, BaseMap, Layer, DatasetSource } from '../../api';
import DeckGL from '@deck.gl/react';
import { MVTLayer } from '@deck.gl/geo-layers';
import { StaticMap } from 'react-map-gl';
import { useDashboard } from '../../Contexts/DashbardBuilderContext';

interface DashboardProps {}

function lookupBaseMapURL(basemap: BaseMap | undefined) {
    switch (basemap) {
        case BaseMap.CartoDBPositron:
            return 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json';
        case BaseMap.CartoDBVoyager:
            return 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json';
        case BaseMap.CartoDBDarkMatter:
            return 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';
        default:
            return 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json';
    }
}

function constructLayer(layer: Layer) {
    const source = layer.source;
    const source_id = Object.values(source)[0];
    let styleType = Object.keys(layer.style)[0];
    let styleSpec = Object.values(layer.style)[0];

    let style = {};
    switch (styleType) {
        case 'Polygon':
            style = {
                getFillColor: styleSpec.fill,
                getBorderColor: styleSpec.stroke,
                stroked: true,
                getLineWidth: styleSpec.stroke_width,
            };
            break;
        case 'Point':
            style = {
                getFillColor: styleSpec.fill,
                getBorderColor: styleSpec.stroke,
                getRadius: styleSpec.size,
            };
            break;
    }

    console.log('Layer name is ', layer.name);

    return new MVTLayer({
        id: layer.name,
        data: `${window.origin}/api/tiler/dataset/${source_id}/{z}/{x}/{y}`,
        ...style,
    });
}
const TOKEN = process.env.REACT_APP_MAPBOX_TOKEN;

export const DashboardViewer: React.FC<DashboardProps> = ({}) => {
    const { dashboard } = useDashboard();

    const INITIAL_VIEW_STATE = {
        longitude: -74.006,
        latitude: 40.7128,
        zoom: 10,
        pitch: 0,
        bearing: 0,
    };

    const mapStyle = dashboard?.map_style;
    const layers = mapStyle
        ? mapStyle.layers.map(constructLayer)
        : [];

    const baseMap = lookupBaseMapURL(mapStyle?.base_map);

    return (
        <Styles.DashboardOuter>
            <DeckGL
                width={'100%'}
                height={'100%'}
                initialViewState={INITIAL_VIEW_STATE}
                layers={layers as any}
                controller={true}
                getTooltip={({ object }: any) => {
                    console.log('tool tip ', object);
                    return object && object.message;
                }}
            >
                <StaticMap
                    mapboxApiAccessToken={TOKEN}
                    width={'100%'}
                    height={'100%'}
                    mapStyle={baseMap}
                />
            </DeckGL>
        </Styles.DashboardOuter>
    );
};
