import NotificationsManager from "@/components/molecules/notifications_manager";

type VertexCredentialsForm = {
  setFieldsValue: (values: { vertex_credentials: string }) => void;
};

type VertexUploadChange = {
  file: {
    name: string;
    status?: string;
  };
};

export function vertexCredentialsUploadProps(form: VertexCredentialsForm) {
  return {
    name: "file",
    accept: ".json",
    pastable: false,
    beforeUpload: (file: File) => {
      if (file.type === "application/json") {
        const reader = new FileReader();
        reader.onload = (event) => {
          if (event.target) {
            form.setFieldsValue({ vertex_credentials: event.target.result as string });
          }
        };
        reader.readAsText(file);
      }
      return false;
    },
    onChange(info: VertexUploadChange) {
      if (info.file.status === "done") {
        NotificationsManager.success(`${info.file.name} file uploaded successfully`);
      } else if (info.file.status === "error") {
        NotificationsManager.fromBackend(`${info.file.name} file upload failed.`);
      }
    },
  };
}
