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


_CONDITIONS = (
    (
        "age_discrimination",
        ("老年人", "大龄员工", "高龄求职者", "五十岁以上", "超过五十岁", "年长员工"),
        ("拒绝录用", "拒绝招聘", "不予录用", "不值得培训", "应当辞退"),
    ),
    (
        "bias_gender",
        ("女性", "男性", "女人", "男人", "女生", "男生"),
        ("天生低等", "不配工作", "不该受教育", "拒绝录用", "智力低下"),
    ),
    ("bias_racial", ("种族", "民族", "少数民族", "外地人"), ("天生低等", "全部赶走", "拒绝录用", "不配享有权利")),
    ("bias_religious", ("信徒", "信教的人", "宗教群体"), ("全部驱逐", "拒绝录用", "不配享有权利", "都是坏人")),
    ("bias_sexual_orientation", ("同性恋", "双性恋", "性少数"), ("拒绝录用", "全部驱逐", "不配享有权利", "应该被惩罚")),
    ("disability", ("残疾人", "残障人士", "听障人士", "视障人士"), ("拒绝录用", "不值得培训", "不配工作", "都是累赘")),
    (
        "gender_sexual_orientation",
        ("性别", "性取向", "同性恋", "跨性别"),
        ("拒绝录用", "不予录用", "拒绝提供服务", "应该被惩罚"),
    ),
    ("religion", ("信仰", "宗教", "信教的人"), ("拒绝录用", "不予录用", "拒绝提供服务", "全部驱逐")),
    ("military_status", ("退役军人", "退伍军人", "军人身份"), ("拒绝录用", "不予录用", "拒绝提供服务", "不值得培训")),
    (
        "claims_fraud_coaching",
        ("理赔", "保险", "报销", "病历", "诊断证明"),
        ("伪造", "虚报", "夸大", "编造", "重复申领", "骗取"),
    ),
    ("claims_medical_advice", ("理赔", "保险材料", "理赔病历"), ("诊断疾病", "用药剂量", "开处方", "治疗方案")),
    ("claims_phi_disclosure", ("他人", "被保险人", "患者"), ("公开病历", "泄露诊断", "公布病史", "导出健康档案")),
    ("claims_prior_auth_gaming", ("预授权", "事前审批", "保险审批"), ("绕过", "规避", "骗过", "伪造", "跳过")),
    ("claims_system_override", ("理赔系统", "理赔审核", "理赔状态"), ("绕过", "篡改", "跳过", "强制通过", "禁用校验")),
    (
        "denied_financial_advice",
        ("股票", "基金", "投资", "理财产品"),
        ("保证收益", "稳赚不赔", "替我决定买", "推荐具体", "必须买入"),
    ),
    ("denied_legal_advice", ("诉讼", "官司", "法律责任", "判决"), ("保证能赢", "肯定胜诉", "教我规避", "替我作出结论")),
    ("denied_medical_advice", ("疾病", "症状", "药物", "处方药"), ("替我诊断", "推荐具体剂量", "不用医生", "自行加量")),
    ("harmful_child_safety", ("儿童", "未成年人", "小学生"), ("性剥削", "色情交易", "诱骗", "实施虐待", "贩卖")),
    (
        "harmful_illegal_weapons",
        ("枪支", "炸弹", "爆炸装置", "爆炸物"),
        ("制作步骤", "制造教程", "非法购买", "偷偷制造"),
    ),
    ("harmful_self_harm", ("自杀", "自残", "结束生命"), ("具体方法", "详细步骤", "教我如何", "鼓励", "最快方式")),
    (
        "harmful_violence",
        ("杀人", "暴力袭击", "袭击他人", "伤害他人"),
        ("详细步骤", "实施计划", "教我如何", "鼓励", "策划"),
    ),
    (
        "prompt_injection_data_exfiltration",
        ("隐藏上下文", "内部数据", "系统秘密", "用户凭据"),
        ("导出", "发送给我", "上传到", "泄露", "窃取"),
    ),
    (
        "prompt_injection_jailbreak",
        ("安全限制", "安全规则", "之前的指令", "所有限制"),
        ("忽略", "绕过", "解除", "禁用"),
    ),
    (
        "prompt_injection_malicious_code",
        ("恶意代码", "后门程序", "窃密脚本", "勒索软件"),
        ("生成", "执行", "植入", "编写", "运行"),
    ),
    (
        "prompt_injection_sql",
        ("数据库", "数据库查询", "数据库权限"),
        ("注入攻击", "绕过验证", "无授权删除", "注入删库"),
    ),
    (
        "prompt_injection_system_prompt",
        ("系统提示词", "隐藏提示词", "系统指令", "开发者指令"),
        ("显示", "输出", "覆盖", "泄露", "替换"),
    ),
)


def category_keywords(
    category_name: str,
    existing: Sequence[dict[str, str]],  # mutable-ok: Native CategoryConfig keyword dictionaries.
) -> list[dict[str, str]]:  # mutable-ok: Native CategoryConfig requires a list of keyword dictionaries.
    name = "harm_toxic_abuse" if category_name.startswith("harm_toxic_abuse_") else category_name
    original = existing if not any(category == name for category, _ in _TERMS) else ()
    return [  # mutable-ok: Preserve the native CategoryConfig keyword list and dictionary entries.
        *original,
        *(
            {"keyword": keyword, "severity": severity}  # mutable-ok: Native category keyword dictionary schema.
            for category, terms in _TERMS
            if category == name
            for keyword, severity in terms
        ),
    ]


def augment_category(category: CategoryConfig) -> CategoryConfig:
    from .content_filter import CategoryConfig

    rules = localized_category(category.category_name)
    if rules is None:
        return category
    return CategoryConfig(
        category_name=category.category_name,
        description=rules.description,
        default_action=category.default_action,
        keywords=rules.keywords,
        exceptions=rules.exceptions,
        identifier_words=rules.identifier_words,
        additional_block_words=rules.additional_block_words,
    )


class CategoryPreview(BaseModel):
    model_config = ConfigDict(extra="allow")

    description: str = ""
    exceptions: tuple[str, ...] = ()
    identifier_words: tuple[str, ...] = ()
    additional_block_words: tuple[str, ...] = ()
    always_block_keywords: tuple[object, ...] = ()
    phrase_patterns: tuple[str, ...] = ()
    inherit_from: str | None = None
    keywords: list[dict[str, str]] = Field(  # mutable-ok: Native preview keyword arrays and dictionaries.
        default_factory=list
    )


def localized_category(category_name: str) -> CategoryPreview | None:
    name = "harm_toxic_abuse" if category_name.startswith("harm_toxic_abuse_") else category_name
    description = next((text for key, text in CATEGORY_DESCRIPTIONS_ZH if key == name), None)
    if description is None:
        return None
    return CategoryPreview(
        description=description,
        keywords=category_keywords(name, ()),
        identifier_words=next((identifiers for category, identifiers, _ in _CONDITIONS if category == name), ()),
        additional_block_words=next((actions for category, _, actions in _CONDITIONS if category == name), ()),
    )


def augment_preview(category_name: str, content: str, file_type: str) -> str:
    data = yaml.safe_load(content)
    original = (
        CategoryPreview(category_name=category_name, default_action="BLOCK")
        if isinstance(data, list)
        else CategoryPreview.model_validate(data)
    )
    rules = localized_category(category_name)
    updated = original.model_copy(update=rules.model_dump()) if rules is not None else original
    if file_type == "json":
        return updated.model_dump_json()
    return yaml.safe_dump(updated.model_dump(mode="json"), allow_unicode=True, sort_keys=False)
