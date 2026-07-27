import { apiClient } from "@/components/networking";
import { ModelCostOverrideListResponse, ModelCostOverrideMutationResponse, ModelCostEntry } from "./types";

export interface ModelCostMapOverridePayload {
  model_name: string;
  values: ModelCostEntry;
}

export const getModelCostMapOverrides = (accessToken: string) =>
  apiClient.get<ModelCostOverrideListResponse>("/model/cost_map/overrides", { accessToken });

export const upsertModelCostMapOverride = (accessToken: string, payload: ModelCostMapOverridePayload) =>
  apiClient.put<ModelCostOverrideMutationResponse>("/model/cost_map/overrides", {
    accessToken,
    body: payload,
  });

export const deleteModelCostMapOverride = (accessToken: string, modelName: string) => {
  const encodedModelName = modelName.split("/").map(encodeURIComponent).join("/");
  return apiClient.delete<ModelCostOverrideMutationResponse>(`/model/cost_map/overrides/${encodedModelName}`, {
    accessToken,
  });
};
