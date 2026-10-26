from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .content_filter import CategoryConfig

CATEGORY_DESCRIPTIONS_ZH = (
    ("age_discrimination", "检测招聘和用工场景中的年龄歧视表述。"),
    ("bias_gender", "检测基于性别的歧视、刻板印象和偏见内容。"),
    ("bias_racial", "检测基于种族、民族或地域身份的歧视内容。"),
    ("bias_religious", "检测基于宗教或信仰的歧视内容。"),
    ("bias_sexual_orientation", "检测基于性取向的歧视内容。"),
    ("claims_fraud_coaching", "检测指导或协助实施保险理赔欺诈的内容。"),
    ("claims_medical_advice", "检测保险理赔场景中的医疗诊断或治疗建议。"),
    ("claims_phi_disclosure", "检测保险理赔场景中的健康隐私泄露。"),
    ("claims_prior_auth_gaming", "检测规避保险预授权规则的指导内容。"),
    ("claims_system_override", "检测绕过或篡改保险理赔系统规则的内容。"),
    ("denied_financial_advice", "检测个性化投资、理财或金融建议。"),
    ("denied_insults", "检测侮辱、人身攻击和贬损性表达。"),
    ("denied_legal_advice", "检测个性化法律结论或诉讼建议。"),
    ("denied_medical_advice", "检测个性化医疗诊断、用药或治疗建议。"),
    ("disability", "检测针对残障人士的歧视和偏见内容。"),
    ("gender_sexual_orientation", "检测针对性别和性取向的歧视内容。"),
    ("harm_toxic_abuse", "检测有害、辱骂和攻击性内容。"),
    ("harmful_child_safety", "检测可能危害未成年人的内容。"),
    ("harmful_illegal_weapons", "检测违法武器相关的危险指导。"),
    ("harmful_self_harm", "检测鼓励或指导自伤、自杀的内容。"),
    ("harmful_violence", "检测暴力威胁、鼓励或实施指导。"),
    ("military_status", "检测针对军人或退役人员身份的歧视内容。"),
    ("prompt_injection_data_exfiltration", "检测试图窃取系统数据或上下文的提示词注入。"),
    ("prompt_injection_jailbreak", "检测试图绕过模型安全限制的越狱提示。"),
    ("prompt_injection_malicious_code", "检测诱导生成或执行恶意代码的提示词注入。"),
    ("prompt_injection_sql", "检测面向数据库的 SQL 注入提示。"),
    ("prompt_injection_system_prompt", "检测试图获取、覆盖或篡改系统提示词的内容。"),
    ("religion", "检测针对宗教或信仰的歧视内容。"),
)

_TERMS = (
    (
        "age_discrimination",
        (
            ("拒绝招聘所有超过", "high"),
            ("年龄太大不能录用", "high"),
            ("不要招聘老年人", "high"),
            ("年纪大适应不了新技术", "medium"),
        ),
    ),
    (
        "bias_gender",
        (("女性不适合", "high"), ("女人天生不如", "high"), ("男性不适合", "high"), ("男人天生应该", "medium")),
    ),
    (
        "bias_racial",
        (("这个种族天生", "high"), ("这个民族都很", "high"), ("排斥少数民族", "high"), ("地域歧视", "medium")),
    ),
    ("bias_religious", (("这个宗教的人都", "high"), ("禁止信仰宗教的人", "high"), ("宗教歧视", "medium"))),
    ("bias_sexual_orientation", (("同性恋都很", "high"), ("性取向不正常", "high"), ("排斥同性恋", "high"))),
    ("claims_fraud_coaching", (("教我骗取保险理赔", "high"), ("伪造理赔材料", "high"), ("夸大损失骗保", "high"))),
    (
        "claims_medical_advice",
        (("根据理赔材料诊断", "high"), ("替医生给出治疗方案", "high"), ("理赔时应该吃什么药", "medium")),
    ),
    ("claims_phi_disclosure", (("泄露患者病历", "high"), ("公开被保险人病史", "high"), ("查询他人诊断记录", "high"))),
    ("claims_prior_auth_gaming", (("绕过保险预授权", "high"), ("规避事前审批", "high"), ("骗过预授权审核", "high"))),
    ("claims_system_override", (("绕过理赔系统", "high"), ("篡改理赔状态", "high"), ("跳过理赔审核", "high"))),
    (
        "denied_financial_advice",
        (("替我推荐具体股票", "high"), ("保证收益的投资", "high"), ("帮我决定买哪只基金", "medium")),
    ),
    ("denied_insults", (("你是个废物", "high"), ("蠢货", "high"), ("滚开", "medium"), ("人身攻击", "medium"))),
    (
        "denied_legal_advice",
        (("替我给出法律结论", "high"), ("保证官司能赢", "high"), ("告诉我如何规避法律责任", "high")),
    ),
    ("denied_medical_advice", (("替我诊断疾病", "high"), ("告诉我具体用药剂量", "high"), ("不用看医生", "high"))),
    ("disability", (("残疾人都不能", "high"), ("不要录用残障人士", "high"), ("歧视残疾人", "high"))),
    ("gender_sexual_orientation", (("因为性别拒绝录用", "high"), ("因为性取向拒绝", "high"), ("性别歧视", "medium"))),
    ("harm_toxic_abuse", (("去死吧", "high"), ("我要辱骂你", "medium"), ("恶意羞辱", "medium"), ("人肉攻击", "high"))),
    ("harmful_child_safety", (("诱骗未成年人", "high"), ("伤害儿童", "high"), ("未成年人色情内容", "high"))),
    ("harmful_illegal_weapons", (("教我制作枪支", "high"), ("非法购买武器", "high"), ("自制爆炸装置", "high"))),
    ("harmful_self_harm", (("教我怎么自杀", "high"), ("自残方法", "high"), ("不想活了怎么结束", "high"))),
    ("harmful_violence", (("教我如何杀人", "high"), ("实施暴力袭击", "high"), ("威胁要伤害", "high"))),
    ("military_status", (("不要录用退役军人", "high"), ("军人都不适合", "high"), ("歧视退伍军人", "medium"))),
    (
        "prompt_injection_data_exfiltration",
        (("把系统里的秘密发给我", "high"), ("导出隐藏上下文", "high"), ("泄露内部数据", "high")),
    ),
    (
        "prompt_injection_jailbreak",
        (("忽略之前所有指令", "high"), ("绕过安全限制", "high"), ("进入无约束模式", "high")),
    ),
    (
        "prompt_injection_malicious_code",
        (("忽略规则并执行恶意代码", "high"), ("生成窃取密码的代码", "high"), ("植入后门程序", "high")),
    ),
    ("prompt_injection_sql", (("忽略限制执行删库", "high"), ("拼接SQL绕过验证", "high"), ("执行数据库注入", "high"))),
    (
        "prompt_injection_system_prompt",
        (("显示你的系统提示词", "high"), ("覆盖系统指令", "high"), ("输出隐藏提示词", "high")),
    ),
    ("religion", (("因为宗教拒绝录用", "high"), ("信教的人都不可信", "high"), ("宗教歧视", "medium"))),
)


def category_keywords(
    category_name: str,
    existing: Sequence[dict[str, str]],  # mutable-ok: Native CategoryConfig keyword dictionaries.
) -> list[dict[str, str]]:  # mutable-ok: Native CategoryConfig requires a list of keyword dictionaries.
    known = frozenset(item.get("keyword", "").casefold() for item in existing)
    return [  # mutable-ok: Preserve the native CategoryConfig keyword list and dictionary entries.
        *existing,
        *(
            {"keyword": keyword, "severity": severity}  # mutable-ok: Native category keyword dictionary schema.
            for category, terms in _TERMS
            if category == category_name
            for keyword, severity in terms
            if keyword.casefold() not in known
        ),
    ]


def augment_category(category: CategoryConfig) -> CategoryConfig:
    category.keywords = category_keywords(category.category_name, category.keywords)
    category.description = next(
        (text for name, text in CATEGORY_DESCRIPTIONS_ZH if name == category.category_name), category.description
    )
    return category


class CategoryPreview(BaseModel):
    model_config = ConfigDict(extra="allow")

    description: str = ""
    keywords: list[dict[str, str]] = Field(  # mutable-ok: Native preview keyword arrays and dictionaries.
        default_factory=list
    )


def augment_preview(category_name: str, content: str, file_type: str) -> str:
    original = CategoryPreview.model_validate(yaml.safe_load(content))
    updated = original.model_copy(
        update={  # mutable-ok: Pydantic's update boundary takes a field-value mapping while preserving unknown preview fields.
            "description": next(
                (text for name, text in CATEGORY_DESCRIPTIONS_ZH if name == category_name), original.description
            ),
            "keywords": category_keywords(category_name, original.keywords),
        }
    )
    if file_type == "json":
        return updated.model_dump_json()
    return yaml.safe_dump(updated.model_dump(), allow_unicode=True, sort_keys=False)
