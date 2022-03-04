import React from "react";
import { Page } from "@maticoapp/matico_spec";
import { Box } from "grommet";
import { MarkdownContent } from "../MarkdownContent/MarkdownContent";
import { MaticoSection } from "../MaticoSection/MaticoSection";
import {
  Tabs,
  View,
  TabList,
  TabPanels,
  Item,
  Flex,
} from "@adobe/react-spectrum";

interface MaticoPageInterface {
  page: Page;
  editPath?: string;
}
export const MaticoPage: React.FC<MaticoPageInterface> = ({
  page,
  editPath,
}) => {
  console.log("Page is ", page);

  let content =
    page.sections.length > 1 ? (
      <Tabs width={"100%"} height={"100%"}>
        <View paddingStart={"size-200"}>
          <TabList>
            {page.sections.map((section: any) => (
              <Item key={section.name}>{section.name}</Item>
            ))}
          </TabList>
        </View>
        <TabPanels>
          {page.sections.map((section, index) => (
            <Item key={section.name}>
              {page.content && (
                <MarkdownContent key="content">{page.content}</MarkdownContent>
              )}

              <MaticoSection
                key={section.name}
                section={section}
                editPath={`${editPath}.sections.${index}`}
              />
            </Item>
          ))}
        </TabPanels>
      </Tabs>
    ) : (
      <Flex direction="column" width={"100%"} height={"100%"}>
        {page.content && (
          <MarkdownContent key="content">{page.content}</MarkdownContent>
        )}

        {page.sections.length === 1 && (
          <MaticoSection
            key={page.sections[0].name}
            section={page.sections[0]}
            editPath={`${editPath}.sections.0`}
          />
        )}
      </Flex>
    );

  return (
    <Box fill={true} overflow={{ vertical: "auto" }}>
      {content}
    </Box>
  );
};
