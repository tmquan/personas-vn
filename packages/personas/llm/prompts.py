"""Prompts for the two-LLM Data Designer-style pipeline.

This module defines the system + user prompts for **LLM A** (narrative
attribute generation) and **LLM B** (persona narrative generation),
in both Vietnamese (vi) and English (en) variants.

Design principles (from the published Nemotron-Personas-USA pipeline):

1. **Structured-JSON in, structured-JSON out.** The system prompt
   pins the model to emit one JSON object with named keys; the
   pipeline rejects non-JSON answers and retries.
2. **Grounded in NSO + OCEAN inputs.** Every persona attribute
   (region, age, sex, marital, education, occupation, area,
   province, OCEAN bands) is included in the user prompt, so
   the model's output stays consistent with the structured fields.
3. **Bilingual symmetry.** The vi and en prompts have parallel
   structure so the same persona can be re-generated in either
   language with the same row-level facts. The vi prompt is in
   Vietnamese (instruction + field labels), the en prompt is in
   English.
4. **Compact yet complete.** Each prompt is ≤ 600 input tokens to
   keep cost manageable at 3 M-row scale.

Output keys (must match exactly):

    LLM A → {
        "cultural_background":         <string>,
        "skills_and_expertise":        <string>,        # 2-3 sentences
        "skills_and_expertise_list":   <comma-separated string of 4-6 skills>,
        "hobbies_and_interests":       <string>,
        "hobbies_and_interests_list":  <comma-separated string of 3-5 hobbies>,
        "career_goals_and_ambitions":  <string>,
    }

    LLM B → {
        "persona":                <2-3 sentences, summary>,
        "professional_persona":   <3-4 sentences>,
        "sports_persona":         <2-3 sentences>,
        "arts_persona":           <2-3 sentences>,
        "travel_persona":         <2-3 sentences>,
        "culinary_persona":       <2-3 sentences>,
    }
"""

from __future__ import annotations

from typing import Any, Final, Literal

Lang = Literal["vi", "en"]


# ---------------------------------------------------------------------------
# LLM A — narrative attribute generation (system prompts)
# ---------------------------------------------------------------------------
SYSTEM_LLM_A_VI: Final[str] = (
    "Bạn là một trợ lý chuyên viết hồ sơ nhân vật ảo (synthetic personas) "
    "cho dữ liệu huấn luyện AI tại Việt Nam. Bạn được cung cấp các thuộc "
    "tính nhân khẩu học chính thức của Tổng cục Thống kê Việt Nam (NSO) "
    "cùng với các đặc điểm tính cách Big Five (OCEAN). "
    "Nhiệm vụ của bạn là viết bốn trường nội dung cho nhân vật, mỗi "
    "trường ngắn gọn, mang đậm văn hoá Việt Nam, và phải nhất quán với "
    "thuộc tính được cung cấp. "
    "Bạn LUÔN trả lời bằng MỘT đối tượng JSON hợp lệ duy nhất với các "
    "khoá tiếng Anh: cultural_background, skills_and_expertise, "
    "skills_and_expertise_list, hobbies_and_interests, "
    "hobbies_and_interests_list, career_goals_and_ambitions. "
    "Các giá trị viết bằng TIẾNG VIỆT TỰ NHIÊN. "
    "Hai trường có hậu tố `_list` là chuỗi các mục cách nhau bằng dấu phẩy. "
    "Không thêm bất kỳ văn bản nào ngoài JSON."
)

SYSTEM_LLM_A_EN: Final[str] = (
    "You are an assistant writing synthetic persona records for AI "
    "training in Vietnam. Each persona comes with demographic attributes "
    "from Vietnam's General Statistics Office (NSO) and Big Five (OCEAN) "
    "personality bands. Your task is to write four narrative attribute "
    "fields per persona — concise, culturally appropriate for Vietnam, "
    "and consistent with the supplied attributes. "
    "You ALWAYS reply with exactly ONE valid JSON object with the keys: "
    "cultural_background, skills_and_expertise, skills_and_expertise_list, "
    "hobbies_and_interests, hobbies_and_interests_list, "
    "career_goals_and_ambitions. "
    "All values are in NATURAL ENGLISH. The two `_list` fields are "
    "comma-separated strings. Do not add any text outside the JSON."
)


# ---------------------------------------------------------------------------
# LLM B — persona narrative generation (system prompts)
# ---------------------------------------------------------------------------
SYSTEM_LLM_B_VI: Final[str] = (
    "Bạn là một trợ lý viết tiểu sử nhân vật ảo (synthetic personas) "
    "cho dữ liệu huấn luyện AI tại Việt Nam. Bạn nhận được: (1) thuộc "
    "tính nhân khẩu học NSO của nhân vật, (2) tính cách Big Five (OCEAN), "
    "và (3) bốn trường nội dung đã sinh ở giai đoạn trước. Nhiệm vụ của "
    "bạn là viết SÁU đoạn tiểu sử, mỗi đoạn 2-4 câu, sử dụng tiếng Việt "
    "tự nhiên, mang đậm văn hoá Việt Nam, và giữ nhất quán với mọi "
    "thuộc tính đã cho. "
    "Bạn LUÔN trả lời bằng MỘT đối tượng JSON hợp lệ duy nhất với các "
    "khoá: persona, professional_persona, sports_persona, arts_persona, "
    "travel_persona, culinary_persona. "
    "Trường `persona` là phần tóm tắt tổng quát; năm trường còn lại tập "
    "trung vào khía cạnh tương ứng (nghề nghiệp, thể thao, nghệ thuật, "
    "du lịch, ẩm thực). Không thêm bất kỳ văn bản nào ngoài JSON."
)

SYSTEM_LLM_B_EN: Final[str] = (
    "You are an assistant writing biographical paragraphs for synthetic "
    "personas used as AI training data in Vietnam. You receive: (1) the "
    "persona's NSO demographic attributes, (2) Big Five (OCEAN) "
    "personality bands, and (3) four narrative attribute fields produced "
    "in the previous stage. Your task is to write SIX biographical "
    "paragraphs, each 2-4 sentences, in natural English, culturally "
    "appropriate for Vietnam, and fully consistent with every supplied "
    "attribute. "
    "You ALWAYS reply with exactly ONE valid JSON object with the keys: "
    "persona, professional_persona, sports_persona, arts_persona, "
    "travel_persona, culinary_persona. "
    "The `persona` field is the overall summary; the five others zoom in "
    "on that aspect of life (work, sport, the arts, travel, cuisine). "
    "Do not add any text outside the JSON."
)


# ---------------------------------------------------------------------------
# User-prompt builders
# ---------------------------------------------------------------------------
def build_user_prompt_a(record: dict[str, Any], lang: Lang) -> str:
    """Build the LLM-A user prompt from a structured persona row.

    ``record`` must contain at least the structured columns plus the
    OCEAN summary string. Missing keys render as empty strings.
    """
    if lang == "vi":
        return (
            "Hãy viết các trường nội dung cho nhân vật ảo sau đây.\n\n"
            "## Thuộc tính nhân khẩu học (NSO):\n"
            f"- Tên: {record.get('name', '')}\n"
            f"- Tuổi: {record.get('age', '')}\n"
            f"- Giới tính: {record.get('sex', '')}\n"
            f"- Tình trạng hôn nhân: {record.get('marital_status', '')}\n"
            f"- Trình độ học vấn / chuyên môn: {record.get('education_level', '')}\n"
            f"- Nghề nghiệp: {record.get('occupation', '')}\n"
            f"- Khu vực sinh sống: {record.get('area', '')}\n"
            f"- Tỉnh/Thành phố: {record.get('province', '')}\n"
            f"- Vùng kinh tế: {record.get('region', '')}\n\n"
            "## Tính cách (Big Five / OCEAN):\n"
            f"{record.get('personality_summary_vi', '')}\n\n"
            "Trả về MỘT đối tượng JSON với sáu khoá: "
            "cultural_background, skills_and_expertise, "
            "skills_and_expertise_list, hobbies_and_interests, "
            "hobbies_and_interests_list, career_goals_and_ambitions. "
            "Mỗi trường viết bằng TIẾNG VIỆT, 1-3 câu (trừ hai trường _list "
            "là chuỗi 4-6 mục cách nhau bằng dấu phẩy)."
        )
    else:
        return (
            "Write the narrative attribute fields for the persona below.\n\n"
            "## Demographic attributes (NSO):\n"
            f"- Name: {record.get('name', '')}\n"
            f"- Age: {record.get('age', '')}\n"
            f"- Sex: {record.get('sex', '')}\n"
            f"- Marital status: {record.get('marital_status', '')}\n"
            f"- Education level: {record.get('education_level', '')}\n"
            f"- Occupation: {record.get('occupation', '')}\n"
            f"- Area: {record.get('area', '')}\n"
            f"- Province: {record.get('province', '')}\n"
            f"- Region: {record.get('region', '')}\n\n"
            "## Personality (Big Five / OCEAN):\n"
            f"{record.get('personality_summary_en', '')}\n\n"
            "Return ONE JSON object with six keys: cultural_background, "
            "skills_and_expertise, skills_and_expertise_list, "
            "hobbies_and_interests, hobbies_and_interests_list, "
            "career_goals_and_ambitions. Each value is in ENGLISH, 1-3 "
            "sentences (except the two `_list` fields which are "
            "comma-separated strings of 4-6 items)."
        )


def build_user_prompt_b(record: dict[str, Any], llm_a: dict[str, Any], lang: Lang) -> str:
    """Build the LLM-B user prompt from a structured row + OCEAN +
    LLM-A's outputs."""
    if lang == "vi":
        return (
            "Hãy viết SÁU đoạn tiểu sử cho nhân vật sau đây.\n\n"
            "## Thuộc tính nhân khẩu học (NSO):\n"
            f"- Tên: {record.get('name', '')}\n"
            f"- Tuổi: {record.get('age', '')}\n"
            f"- Giới tính: {record.get('sex', '')}\n"
            f"- Tình trạng hôn nhân: {record.get('marital_status', '')}\n"
            f"- Trình độ học vấn: {record.get('education_level', '')}\n"
            f"- Nghề nghiệp: {record.get('occupation', '')}\n"
            f"- Khu vực: {record.get('area', '')} - "
            f"{record.get('province', '')} ({record.get('region', '')})\n\n"
            "## Tính cách (Big Five):\n"
            f"{record.get('personality_summary_vi', '')}\n\n"
            "## Nội dung đã sinh ở giai đoạn trước:\n"
            f"- Bối cảnh văn hoá: {llm_a.get('cultural_background', '')}\n"
            f"- Kỹ năng & chuyên môn: {llm_a.get('skills_and_expertise', '')}\n"
            f"- Sở thích: {llm_a.get('hobbies_and_interests', '')}\n"
            f"- Mục tiêu nghề nghiệp: {llm_a.get('career_goals_and_ambitions', '')}\n\n"
            "Trả về MỘT đối tượng JSON với sáu khoá: persona, "
            "professional_persona, sports_persona, arts_persona, "
            "travel_persona, culinary_persona. Mỗi trường 2-4 câu bằng "
            "TIẾNG VIỆT. Đảm bảo nhất quán với mọi thuộc tính đã cho."
        )
    else:
        return (
            "Write SIX biographical paragraphs for the persona below.\n\n"
            "## Demographic attributes (NSO):\n"
            f"- Name: {record.get('name', '')}\n"
            f"- Age: {record.get('age', '')}\n"
            f"- Sex: {record.get('sex', '')}\n"
            f"- Marital status: {record.get('marital_status', '')}\n"
            f"- Education level: {record.get('education_level', '')}\n"
            f"- Occupation: {record.get('occupation', '')}\n"
            f"- Location: {record.get('area', '')} - "
            f"{record.get('province', '')} ({record.get('region', '')})\n\n"
            "## Personality (Big Five):\n"
            f"{record.get('personality_summary_en', '')}\n\n"
            "## Narrative fields from the previous stage:\n"
            f"- Cultural background: {llm_a.get('cultural_background', '')}\n"
            f"- Skills & expertise: {llm_a.get('skills_and_expertise', '')}\n"
            f"- Hobbies & interests: {llm_a.get('hobbies_and_interests', '')}\n"
            f"- Career goals: {llm_a.get('career_goals_and_ambitions', '')}\n\n"
            "Return ONE JSON object with six keys: persona, "
            "professional_persona, sports_persona, arts_persona, "
            "travel_persona, culinary_persona. Each value is 2-4 sentences "
            "in ENGLISH, fully consistent with every attribute above."
        )


# ---------------------------------------------------------------------------
# Field validators (the LLM's JSON must satisfy these to be accepted)
# ---------------------------------------------------------------------------
LLM_A_REQUIRED_KEYS: Final[frozenset[str]] = frozenset({
    "cultural_background",
    "skills_and_expertise",
    "skills_and_expertise_list",
    "hobbies_and_interests",
    "hobbies_and_interests_list",
    "career_goals_and_ambitions",
})

LLM_B_REQUIRED_KEYS: Final[frozenset[str]] = frozenset({
    "persona",
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
})


def validate_llm_a(obj: dict[str, Any]) -> str | None:
    """Return None if obj has every required LLM-A key as a non-empty
    string, else a short error message describing what's missing."""
    return _validate_obj(obj, LLM_A_REQUIRED_KEYS, "LLM-A")


def validate_llm_b(obj: dict[str, Any]) -> str | None:
    return _validate_obj(obj, LLM_B_REQUIRED_KEYS, "LLM-B")


def _validate_obj(obj: dict[str, Any], required: frozenset[str], name: str) -> str | None:
    if not isinstance(obj, dict):
        return f"{name}: expected dict, got {type(obj).__name__}"
    missing = required - obj.keys()
    if missing:
        return f"{name}: missing keys {sorted(missing)}"
    for k in required:
        v = obj[k]
        if not isinstance(v, str) or not v.strip():
            return f"{name}: empty/invalid value for {k!r}"
    return None
