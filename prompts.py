DESCRIPTION_GUIDELINES = (
    "Focus your description on the observable video content. Cover the primary "
    "subjects, their actions and transitions, the scene environment and noteworthy "
    "background elements, the lighting and color palette, the camera motion or "
    "framing, the pacing of events, and the overall atmosphere or mood. "
    "Write in the present tense as a single cohesive paragraph without bullet points."
)


def _kw_reference():
    return (
        "Extract key visual elements from this video as a list of keywords. "
        "Include only the most important elements: main subjects, key actions, "
        "scene type, lighting style, camera movement. "
        'Format as comma-separated keywords (e.g., "person walking, urban street, '
        'daytime, handheld camera, busy atmosphere"). '
        "Keep it concise - 8-12 keywords maximum."
    )


def _kw_edit(edit_instruction):
    return (
        "Apply the following edit instructions to the reference video and "
        "extract key visual elements from the edited video.\n\n"
        f"Edit instructions:\n{edit_instruction}\n\n"
        "List the key visual elements as comma-separated keywords focusing on: "
        "main subjects, key actions, scene type, lighting style, camera movement, "
        "atmosphere.\nKeep it concise - 8-12 keywords maximum.\n"
        "Reflect the changes from the edit instruction in the keywords."
    )


def _kw_edit_reasoning(edit_instruction):
    return (
        f'Analyze how this edit instruction changes the reference video: "{edit_instruction}"\n\n'
        "Identify in 2-3 sentences:\n"
        "1. What specific visual elements are being modified or replaced\n"
        "2. The key visual differences between before and after the edit\n\n"
        "Be concrete and focus only on the changed elements, not unchanged background details."
    )


def _kw_edit_description(edit_instruction, reasoning_trace):
    return (
        "Extract key visual elements from the edited video as a list of keywords.\n\n"
        f'Edit applied: "{edit_instruction}"\n\n'
        f"Based on this analysis:\n{reasoning_trace}\n\n"
        "List the key visual elements as comma-separated keywords focusing on: "
        "main subjects, key actions, scene type, lighting style, camera movement, "
        "atmosphere.\nKeep it concise - 8-12 keywords maximum.\n"
        "Reflect the changes from the edit instruction in the keywords."
    )


def _dense_reference():
    return f"Describe the contents of this reference video in vivid detail. {DESCRIPTION_GUIDELINES}"


def _dense_edit(edit_instruction):
    return (
        "Apply the following edit instructions to the reference video and "
        f"describe the resulting edited video in vivid detail.\n{DESCRIPTION_GUIDELINES}\n"
        f"Edit instructions:\n{edit_instruction}"
    )


def _dense_edit_reasoning(edit_instruction):
    return (
        "Given this reference video, analyze the following edit instruction "
        "and reason about how the video would change.\n\n"
        f"Edit instruction:\n{edit_instruction}\n\n"
        "Provide a concise reasoning trace that covers:\n"
        "1. What key elements in the reference video would be affected\n"
        "2. What specific transformations would occur\n"
        "3. What the result would look like after applying the edit\n\n"
        "Keep your reasoning focused and concrete."
    )


def _dense_edit_description(edit_instruction, reasoning_trace):
    return (
        "Based on the reasoning below, describe the edited video in vivid detail.\n"
        f"{DESCRIPTION_GUIDELINES}\n\n"
        f"Edit instruction:\n{edit_instruction}\n\n"
        f"Reasoning:\n{reasoning_trace}\n\n"
        "Now provide a detailed description of the edited video:"
    )


def _reason_reference():
    return f"Describe the contents of this reference video in vivid detail. {DESCRIPTION_GUIDELINES}"


def _reason_edit(edit_instruction):
    return f"Describe the video after this edit: {edit_instruction}"


def _reason_edit_reasoning(edit_instruction):
    return (
        "Given this reference video and the edit instruction below, "
        "briefly think through what the edited video would look like.\n\n"
        f"Edit instruction:\n{edit_instruction}\n\n"
        "Describe only the key changes that would occur:"
    )


def _reason_edit_description(edit_instruction, reasoning_trace):
    return (
        f"You previously identified these changes: {reasoning_trace}\n\n"
        "Now, apply the edit instruction to the reference video and "
        f"describe the resulting edited video in vivid detail.\n{DESCRIPTION_GUIDELINES}\n\n"
        f"Edit instruction:\n{edit_instruction}"
    )


def _strict_reference():
    return (
        "Describe only the visible content of this video as one short paragraph. "
        "Include the main subject, the main action, the scene, the lighting or color, "
        "the camera or framing, and the tempo if visible. "
        "Do not mention the user, the prompt, the task, frames, timestamps, analysis, "
        "reasoning, bullet points, section headers, or markdown."
    )


def _strict_edit(edit_instruction):
    return (
        "Describe the edited target video as one short paragraph.\n\n"
        f"Edit instruction:\n{edit_instruction}\n\n"
        "Describe only observable target-video content. "
        "Do not mention the user, the prompt, the edit instruction, analysis, reasoning, "
        "frames, timestamps, bullet points, section headers, or markdown."
    )


def _strict_edit_reasoning(edit_instruction):
    return (
        "Infer the edited target video from the reference video and the edit instruction.\n\n"
        f"Edit instruction:\n{edit_instruction}\n\n"
        "Output exactly five lines in this format and nothing else:\n"
        "states: ...\n"
        "actions: ...\n"
        "scene: ...\n"
        "camera: ...\n"
        "tempo: ..."
    )


def _strict_edit_description(edit_instruction, reasoning_trace):
    return (
        "Use the reference video and the structured notes below to describe the edited target video.\n\n"
        f"Edit instruction:\n{edit_instruction}\n\n"
        f"Structured notes:\n{reasoning_trace}\n\n"
        "Output exactly one short paragraph describing only the observable edited video. "
        "Do not mention the user, the prompt, the edit instruction, structured notes, analysis, "
        "reasoning, frames, timestamps, bullet points, section headers, or markdown."
    )


def _transition_reference():
    return (
        "Describe the visible reference video using the exact fields below and nothing else.\n"
        "Keep each field compact and retrieval-oriented.\n\n"
        "summary: ...\n"
        "object_primary: ...\n"
        "object_secondary: ...\n"
        "action_chain: ...\n"
        "end_state: ...\n"
        "scene_start: ...\n"
        "scene_end: ...\n"
        "scene_transition: ...\n"
        "camera_transition: ...\n"
        "hand_interaction: ...\n"
        "distractors: ...\n\n"
        "Rules:\n"
        "- object_primary should name the main non-hand object or the main moving actor; use hand as primary only if no other salient object exists\n"
        "- object_secondary may include the hand only when the hand matters for retrieval\n"
        "- action_chain must preserve temporal order using ` -> ` between steps\n"
        "- if the video changes stage, scene, or viewpoint, preserve that change explicitly\n"
        "- end_state must describe the final visible settled condition\n"
        "- omit style words such as calm, clean, deliberate, focused, simple, cozy\n"
        "- do not invent hidden intent or off-screen events"
    )


def _exact_slots_reference():
    return (
        "Describe the visible video for retrieval using exactly these fields and nothing else.\n"
        "summary: ...\n"
        "object_primary: ...\n"
        "object_attributes: exact color/material/shape/count/text/logo/brand if visible\n"
        "object_secondary: ...\n"
        "surface_and_scene: exact surface material plus background room/context\n"
        "action_chain: ordered visible steps using ` -> `; write `none` if static\n"
        "hand_interaction: hand count, hand side/pose, verb, contacted object, no contact if none\n"
        "end_state: final visible settled condition after the main action\n"
        "scene_transition: internal scene/view change or `none`\n"
        "camera_transition: pan/zoom/cut/viewpoint change or `none`\n"
        "distractors: visible but noncentral objects\n\n"
        "Rules:\n"
        "- Never replace a specific object with a generic phrase if the object is visually identifiable.\n"
        "- Preserve readable text, labels, logos, brands, and printed titles exactly when visible.\n"
        "- Distinguish place/grasp/lift/press/point/pour/roll/open/close/reveal/collapse.\n"
        "- Distinguish static objects from moving objects and hand-driven actions.\n"
        "- Do not infer hidden intent; describe only visible evidence."
    )


def _transition_edit(edit_instruction):
    return (
        "Describe the edited target video using the exact fields below and nothing else.\n\n"
        f"Edit instruction:\n{edit_instruction}\n\n"
        "summary: ...\n"
        "object_primary: ...\n"
        "object_secondary: ...\n"
        "action_chain: ...\n"
        "end_state: ...\n"
        "scene_start: ...\n"
        "scene_end: ...\n"
        "scene_transition: ...\n"
        "camera_transition: ...\n"
        "hand_interaction: ...\n\n"
        "Rules:\n"
        "- object_primary should name the main non-hand target object or the main moving actor; use hand as primary only if no other salient object exists\n"
        "- put hand-specific information in hand_interaction instead of making hand the main subject by default\n"
        "- preserve temporal order with ` -> `\n"
        "- describe only the final edited target video, not the source video and not the source-to-target replacement process\n"
        "- keep any required start-to-end scene or viewpoint change explicit\n"
        "- if the edited target stays in one scene or one view, write `scene_transition: none` and `camera_transition: none`\n"
        "- do not keep removed source-side content as if it were still central to the target"
    )


def _transition_edit_reasoning(edit_instruction):
    return (
        "Infer the edited target video from the reference video and the edit instruction.\n\n"
        f"Edit instruction:\n{edit_instruction}\n\n"
        "Output exactly the following fields and nothing else:\n"
        "keep: ...\n"
        "remove: ...\n"
        "replace_with: ...\n"
        "action_chain_after: ...\n"
        "end_state_after: ...\n"
        "scene_start_after: ...\n"
        "scene_end_after: ...\n"
        "scene_transition_after: ...\n"
        "camera_transition_after: ...\n"
        "hand_interaction_after: ...\n"
        "forbidden: ...\n\n"
        "Rules:\n"
        "- action_chain_after and object fields must describe the final edited target video, not the source video\n"
        "- prefer the main non-hand object or main moving actor as the target anchor; use hand as primary only if no other salient object exists\n"
        "- action_chain_after must keep the ordered visible steps using ` -> `\n"
        "- describe only the final edited target video timeline, not the source-to-target replacement process\n"
        "- if the target begins in one view or scene and ends in another, state both\n"
        "- if the target does not change scene or viewpoint internally, write `scene_transition_after: none` and `camera_transition_after: none`\n"
        "- forbidden should keep removed source content explicit\n"
        "- avoid generic style or quality language"
    )


def _transition_edit_description(edit_instruction, reasoning_trace):
    return (
        "Use the reference video and the structured edit notes below to describe the edited target video.\n\n"
        f"Edit instruction:\n{edit_instruction}\n\n"
        f"Structured edit notes:\n{reasoning_trace}\n\n"
        "Output exactly the following fields and nothing else:\n"
        "summary: ...\n"
        "object_primary: ...\n"
        "object_secondary: ...\n"
        "action_chain: ...\n"
        "end_state: ...\n"
        "scene_start: ...\n"
        "scene_end: ...\n"
        "scene_transition: ...\n"
        "camera_transition: ...\n"
        "hand_interaction: ...\n\n"
        "Rules:\n"
        "- object_primary should name the main non-hand target object or the main moving actor; use hand as primary only if no other salient object exists\n"
        "- put hand-specific details in hand_interaction instead of making hand the main subject by default\n"
        "- preserve ordered action steps using ` -> `\n"
        "- describe only the final edited target video timeline, not the source-to-target replacement process\n"
        "- preserve multi-stage scene or viewpoint changes instead of flattening them into one static summary\n"
        "- if the target stays in one scene or viewpoint, write `scene_transition: none` and `camera_transition: none`\n"
        "- end_state must describe the final visible condition after the main action\n"
        "- do not reintroduce removed source-side content as if it were still part of the target"
    )


def _slots_reference():
    return (
        "Describe the visible video content using the exact fields below and nothing else:\n"
        "object_primary: ...\n"
        "object_secondary: ...\n"
        "action_chain: ...\n"
        "surface: ...\n"
        "background: ...\n"
        "color_material: ...\n"
        "hand_interaction: ...\n"
        "end_state: ...\n"
        "distractors: ..."
    )


def _slots_edit(edit_instruction):
    return (
        "Describe the edited target video using the exact fields below and nothing else.\n\n"
        f"Edit instruction:\n{edit_instruction}\n\n"
        "summary: ...\n"
        "object_primary: ...\n"
        "object_secondary: ...\n"
        "action_chain: ...\n"
        "surface: ...\n"
        "background: ...\n"
        "color_material: ...\n"
        "hand_interaction: ...\n"
        "end_state: ...\n"
        "forbidden: ..."
    )


def _slots_edit_reasoning(edit_instruction):
    return (
        "Infer the edited target video from the reference video and the edit instruction.\n\n"
        f"Edit instruction:\n{edit_instruction}\n\n"
        "Output exactly the following fields and nothing else:\n"
        "keep: ...\n"
        "remove: ...\n"
        "replace_with: ...\n"
        "action_after: ...\n"
        "surface_after: ...\n"
        "background_after: ...\n"
        "color_material_after: ...\n"
        "hand_after: ...\n"
        "end_state_after: ...\n"
        "forbidden: ..."
    )


def _slots_edit_description(edit_instruction, reasoning_trace):
    return (
        "Use the reference video and the edit notes below to describe the edited target video.\n\n"
        f"Edit instruction:\n{edit_instruction}\n\n"
        f"Edit notes:\n{reasoning_trace}\n\n"
        "Output exactly the following fields and nothing else:\n"
        "summary: ...\n"
        "object_primary: ...\n"
        "object_secondary: ...\n"
        "action_chain: ...\n"
        "surface: ...\n"
        "background: ...\n"
        "color_material: ...\n"
        "hand_interaction: ...\n"
        "end_state: ...\n"
        "forbidden: ..."
    )


def _default_synthesis(edit_instruction, reasoning_traces):
    traces_text = "\n\n".join(
        [f"Analysis {i + 1}: {trace}" for i, trace in enumerate(reasoning_traces)]
    )
    return (
        f'Edit instruction: "{edit_instruction}"\n\n'
        f"Multiple analyses:\n{traces_text}\n\n"
        "Synthesize these into one consistent, concise understanding "
        "(2-3 sentences) of the key visual changes."
    )


def _reason_synthesis(edit_instruction, reasoning_traces):
    traces_text = "\n\n".join(
        [f"Perspective {i + 1}: {trace}" for i, trace in enumerate(reasoning_traces)]
    )
    return (
        f"Given this edit instruction: {edit_instruction}\n\n"
        f"Multiple perspectives on how the video would change:\n{traces_text}\n\n"
        "Synthesize these perspectives into a single, consistent understanding of the key changes:"
    )


PROMPT_STYLES = {
    "keyword": {
        "reference": _kw_reference,
        "edit": _kw_edit,
        "edit_reasoning": _kw_edit_reasoning,
        "edit_description": _kw_edit_description,
        "consistency_synthesis": _default_synthesis,
    },
    "dense": {
        "reference": _dense_reference,
        "edit": _dense_edit,
        "edit_reasoning": _dense_edit_reasoning,
        "edit_description": _dense_edit_description,
        "consistency_synthesis": _default_synthesis,
    },
    "reasoning": {
        "reference": _reason_reference,
        "edit": _reason_edit,
        "edit_reasoning": _reason_edit_reasoning,
        "edit_description": _reason_edit_description,
        "consistency_synthesis": _reason_synthesis,
    },
    "reasoning_strict": {
        "reference": _strict_reference,
        "edit": _strict_edit,
        "edit_reasoning": _strict_edit_reasoning,
        "edit_description": _strict_edit_description,
        "consistency_synthesis": _reason_synthesis,
    },
    "reasoning_transition_strict": {
        "reference": _transition_reference,
        "edit": _transition_edit,
        "edit_reasoning": _transition_edit_reasoning,
        "edit_description": _transition_edit_description,
        "consistency_synthesis": _reason_synthesis,
    },
    "reasoning_exact_slots_gallery": {
        "reference": _exact_slots_reference,
        "edit": _transition_edit,
        "edit_reasoning": _transition_edit_reasoning,
        "edit_description": _transition_edit_description,
        "consistency_synthesis": _reason_synthesis,
    },
    "reasoning_slots": {
        "reference": _slots_reference,
        "edit": _slots_edit,
        "edit_reasoning": _slots_edit_reasoning,
        "edit_description": _slots_edit_description,
        "consistency_synthesis": _reason_synthesis,
    },
}


def get_prompts(style: str) -> dict:
    return PROMPT_STYLES[style]
