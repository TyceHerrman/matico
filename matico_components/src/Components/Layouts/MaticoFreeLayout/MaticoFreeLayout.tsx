import React from 'react'
import styled from 'styled-components'
import {Box} from 'grommet'
import { PanePosition} from 'matico_spec'
import Draggable from "react-draggable";

const FreeArea = styled.div`
  position: relative;
  width:100%;
  height:100%;
  flex:1;
`

const FreePane = styled.div<{pane:PanePosition}>`
  position: absolute;
  width:${({pane})=> `${pane.width}%`};
  height:${({pane})=> `${pane.height}%`};
  z-index:${({pane})=> `${pane.layer}`};
  left:${({pane})=> `${pane.x}%`};
  bottom:${({pane})=>`${pane.y}%`};
  cursor:${({pane})=> pane.float ? 'grab' : 'pointer'};
  background:${({pane})=> pane.background ? pane.background : 'white'};
`

interface MaticoFreeLayoutInterface{

}

export const MaticoFreeLayout: React.FC<MaticoFreeLayoutInterface> =({children})=>{
  return <FreeArea>
    {React.Children.map(children, child=>{
      //@ts-ignore
      //TODO Make this properly typed. Properly check to ensure that the child nodes implement MaticoPaneInterface 
      const pane  =(<FreePane key={child.props.name} pane={child.props.position} className="FreePane">
        {child}
        </FreePane>)

        //@ts-ignore
        return  child.props.float ? <Draggable>{pane}</Draggable> : pane
        
    })}
  </FreeArea>
}
