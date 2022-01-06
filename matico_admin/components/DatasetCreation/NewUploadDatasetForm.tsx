import {
  View,
  IllustratedMessage,
  Heading,
  Content,
  Tabs,
  TabList,
  TabPanels,
  Item,
  Flex,
} from "@adobe/react-spectrum";
import React, { useCallback, useState } from "react";
import Upload from "@spectrum-icons/illustrations/Upload";
import { useDropzone } from "react-dropzone";
import { FilePreviewer } from "./FilePreviewer";
import { faAlignJustify } from "@fortawesome/free-solid-svg-icons";

export interface NewUploadDatasetFormProps {}

export const NewUploadDatasetForm: React.FC<NewUploadDatasetFormProps> = () => {
  const [acceptedFiles, setAcceptedFiles] = useState<Array<File> | null>(null);

  const onDrop = useCallback((acceptedFiles: Array<File>) => {
    console.log("acceptedFiles ", acceptedFiles);
    setAcceptedFiles(acceptedFiles);
  }, []);

  const { getRootProps, getInputProps, isDragActive, isDragReject } =
    useDropzone({
      onDrop,
      accept:
        "application/geo+json,application/json,application/csv,text/csv,text/plain,*.csv,*.geojson",
    });

  const dropMessage = isDragActive ? "Drop it here!" : "Drag and drop a file";
  const message = isDragReject
    ? "Only csv, json and geojson files are currently supported"
    : dropMessage;

  return (
    <View paddingTop="size-400" height="100%">
      {!acceptedFiles && (
        <Flex alignItems="center" justifyContent="center" height="100%">
          <View>
            <div {...getRootProps()}>
              <input {...getInputProps()} />
              <IllustratedMessage>
                <Upload />
                <Heading>{message}</Heading>
                <Content>Select a File from your computer</Content>
              </IllustratedMessage>
            </div>
          </View>
        </Flex>
      )}
      {acceptedFiles && (
        <Tabs orientation="vertical" width="100%" height="100%">
          <TabList>
            {acceptedFiles.map((file: File) => (
              <Item key={file.name}>{file.name}</Item>
            ))}
          </TabList>
          <Flex width="100%" height="100%">
            <TabPanels>
              {acceptedFiles.map((file: File) => (
                <Item key={file.name}>
                  <FilePreviewer file={file} />
                </Item>
              ))}
            </TabPanels>
          </Flex>
        </Tabs>
      )}
    </View>
  );
};
