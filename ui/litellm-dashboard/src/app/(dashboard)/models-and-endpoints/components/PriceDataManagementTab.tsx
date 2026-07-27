import { TabPanel } from "@tremor/react";
import React from "react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useModelCostMap } from "../../hooks/models/useModelCostMap";
import PricingRulesManager from "./pricing-rules/PricingRulesManager";

const PriceDataManagementTab = () => {
  const { accessToken, userRole } = useAuthorized();
  const { data: modelCostMap, isLoading, refetch: refetchModelCostMap } = useModelCostMap();

  return (
    <TabPanel>
      <div className="p-6">
        <PricingRulesManager
          accessToken={accessToken}
          userRole={userRole}
          modelCostMap={modelCostMap ?? {}}
          loading={isLoading}
          onModelCostMapReload={() => void refetchModelCostMap()}
        />
      </div>
    </TabPanel>
  );
};

export default PriceDataManagementTab;
