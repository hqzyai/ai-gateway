import { fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Form } from "antd";
import { describe, expect, it } from "vitest";
import { getPlaceholder, Providers } from "../provider_info_helpers";
import LiteLLMModelNameField from "./litellm_model_name";

describe("LitellmModelNameField", () => {
  it("should render", () => {
    const { getByText } = render(
      <Form>
        <LiteLLMModelNameField
          selectedProvider={Providers.OpenAI}
          providerModels={[]}
          getPlaceholder={getPlaceholder}
        />
      </Form>,
    );
    expect(getByText("LiteLLM Model Name(s)")).toBeInTheDocument();
  });

  it("should show Azure placeholder as 'my-deployment'", () => {
    const { getByPlaceholderText, queryByPlaceholderText } = render(
      <Form>
        <LiteLLMModelNameField selectedProvider={Providers.Azure} providerModels={[]} getPlaceholder={getPlaceholder} />
      </Form>,
    );
    expect(getByPlaceholderText("my-deployment")).toBeInTheDocument();
    expect(queryByPlaceholderText("gpt-3.5-turbo")).toBeNull();
  });

  it("should add a manually entered model and create its mapping", async () => {
    const { result } = renderHook(() => Form.useForm());
    const [form] = result.current;
    const user = userEvent.setup();

    render(
      <Form form={form}>
        <LiteLLMModelNameField
          selectedProvider={Providers.OpenAI}
          providerModels={["openai/gpt-4.1"]}
          getPlaceholder={getPlaceholder}
        />
      </Form>,
    );

    const combobox = screen.getByRole("combobox");
    await user.type(combobox, "openai/custom-model");
    fireEvent.keyDown(combobox, { key: "Enter", code: "Enter", keyCode: 13 });

    await waitFor(() => {
      expect(form.getFieldValue("model")).toEqual(["openai/custom-model"]);
      expect(form.getFieldValue("model_mappings")).toEqual([
        {
          public_name: "openai/custom-model",
          litellm_model: "openai/custom-model",
        },
      ]);
    });
  });
});
