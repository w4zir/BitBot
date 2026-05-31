from __future__ import annotations



import json

import re

from pathlib import Path



import pytest



from testing.simulator.config import SeedConfig

from testing.scripts.generate_category_intent_dataset import (

    GenerationResult,

    _build_parser,

    _count_jsonl_rows,

    _default_output_path,

    build_fake_scenario_instance,

    run_generation,

)





def _write_minimal_simulator_tree(

    root: Path,

    *,

    suite_yaml: str,

    seed_yaml: str,

    persona_yaml: str,

) -> Path:

    (root / "seeds").mkdir(parents=True)

    (root / "personas").mkdir(parents=True)

    (root / "suites").mkdir(parents=True)

    (root / "suites" / "mini.yaml").write_text(suite_yaml, encoding="utf-8")

    (root / "seeds" / "mini.yaml").write_text(seed_yaml, encoding="utf-8")

    (root / "personas" / "personas.yaml").write_text(persona_yaml, encoding="utf-8")

    return root / "suites" / "mini.yaml"





MINI_SEEDS = """

seeds:

  - seed_id: s1

    category: order

    intent: cancel_order

    description: "Cancel"

    db_filter:

      entity_type: order

      order_status: [processing]

  - seed_id: s2

    category: refund

    intent: get_refund

    description: "Refund"

    db_filter:

      entity_type: order

      order_status: [delivered]

"""



MINI_PERSONAS = """

personas:

  - persona_id: polite_first_timer

    display_name: Polite

    vocabulary: simple

    patience: high

    cooperation_level: cooperative

    escalation_tendency: low

    typical_message_length: medium

    traits: []

  - persona_id: policy_prober

    display_name: Prober

    vocabulary: technical

    patience: medium

    cooperation_level: passive

    escalation_tendency: medium

    typical_message_length: medium

    traits: []

"""





def test_build_fake_scenario_instance_order_user_subscription() -> None:

    order_seed = SeedConfig.model_validate(

        {

            "seed_id": "order_seed",

            "category": "order",

            "intent": "cancel_order",

            "description": "d",

            "db_filter": {"entity_type": "order", "order_status": ["shipped"]},

        }

    )

    o1 = build_fake_scenario_instance(order_seed, 1)

    o2 = build_fake_scenario_instance(order_seed, 2)

    assert re.fullmatch(r"ORD-\d{4}", o1.entity["order_id"])

    assert re.fullmatch(r"ORD-\d{4}", o2.entity["order_id"])

    assert o1.entity["status"] == "shipped"



    user_seed = SeedConfig.model_validate(

        {

            "seed_id": "user_seed",

            "category": "invoice",

            "intent": "check_invoice",

            "description": "d",

            "db_filter": {"entity_type": "user", "user_status": ["active"]},

        }

    )

    u = build_fake_scenario_instance(user_seed, 3)

    assert u.entity["entity_type"] == "user"

    email = str(u.entity.get("email") or "")

    assert re.fullmatch(r"customer\d{4}@example\.test", email)



    sub_seed = SeedConfig.model_validate(

        {

            "seed_id": "sub_seed",

            "category": "subscription",

            "intent": "subscription_status",

            "description": "d",

            "db_filter": {

                "entity_type": "subscription",

                "subscription_status": ["active"],

                "subscription_plan": ["pro"],

            },

        }

    )

    s = build_fake_scenario_instance(sub_seed, 4)

    assert s.entity["entity_type"] == "subscription"

    acct = str(s.entity.get("account_email") or "")

    assert re.fullmatch(r"subscriber\d{4}@example\.test", acct)

    assert s.entity.get("plan") == "pro"





def test_resolve_output_dir_relative_to_repo() -> None:

    from testing.scripts import generate_category_intent_dataset as mod



    resolved = mod._resolve_output_dir(Path("data") / "raw" / "simulated")

    assert resolved.name == "simulated"

    assert "data" in resolved.parts and "raw" in resolved.parts





def test_default_output_path_prefix_and_suffix(tmp_path: Path) -> None:

    p = _default_output_path(tmp_path)

    assert p.parent == tmp_path

    assert p.name.startswith("category_intent_")

    assert p.suffix == ".jsonl"





def test_run_generation_writes_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    sim_root = tmp_path / "simulator"

    suite_path = _write_minimal_simulator_tree(

        sim_root,

        suite_yaml="""

run_id: mini

scenarios:

  - seed_id: s1

    persona_id: polite_first_timer

  - seed_id: s2

    persona_id: policy_prober

""",

        seed_yaml=MINI_SEEDS,

        persona_yaml=MINI_PERSONAS,

    )



    from testing.simulator import persona as persona_mod



    monkeypatch.setattr(

        persona_mod.PersonaEngine,

        "generate_opening",

        lambda self: f"Need help with {self.scenario.entity.get('order_id')} for {self.scenario.seed_id}",

    )



    out = tmp_path / "out.jsonl"

    result = run_generation(

        output_path=out,

        max_limit=3,

        suite_path=suite_path,

        seed_override=None,

        persona_filters=[],

        randomize=False,

        stderr_print=lambda _m: None,

        max_attempt_multiplier=100,

    )

    assert isinstance(result, GenerationResult)

    assert result.rows_written == 3

    assert out.exists()

    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(lines) == 3

    for row in lines:

        assert set(row.keys()) == {"text", "category", "intent", "seed_id", "persona"}

        assert re.search(r"ORD-\d{4}", row["text"])

    assert lines[0]["category"] == "order" and lines[0]["intent"] == "cancel_order"

    assert lines[0]["seed_id"] == "s1" and lines[0]["persona"] == "polite_first_timer"

    assert lines[1]["category"] == "refund" and lines[1]["intent"] == "get_refund"

    assert lines[1]["seed_id"] == "s2" and lines[1]["persona"] == "policy_prober"

    assert lines[2]["category"] == "order" and lines[2]["intent"] == "cancel_order"

    assert lines[2]["seed_id"] == "s1" and lines[2]["persona"] == "polite_first_timer"





def test_run_generation_seed_override_filters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    sim_root = tmp_path / "simulator"

    suite_path = _write_minimal_simulator_tree(

        sim_root,

        suite_yaml="""

run_id: mini

scenarios:

  - seed_id: s1

  - seed_id: s2

""",

        seed_yaml=MINI_SEEDS,

        persona_yaml=MINI_PERSONAS,

    )



    from testing.simulator import persona as persona_mod



    monkeypatch.setattr(

        persona_mod.PersonaEngine,

        "generate_opening",

        lambda self: f"Msg {self.scenario.seed_id} {self.scenario.entity.get('order_id')}",

    )



    out = tmp_path / "out.jsonl"

    result = run_generation(

        output_path=out,

        max_limit=2,

        suite_path=suite_path,

        seed_override="s2",

        persona_filters=[],

        randomize=False,

        stderr_print=lambda _m: None,

    )

    assert result.rows_written == 2

    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert all(row["intent"] == "get_refund" for row in lines)

    for row in lines:

        assert re.search(r"ORD-\d{4}", row["text"])





def test_run_generation_persona_filter_must_match_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    sim_root = tmp_path / "simulator"

    suite_path = _write_minimal_simulator_tree(

        sim_root,

        suite_yaml="""

run_id: mini

scenarios:

  - seed_id: s1

    persona_id: policy_prober

""",

        seed_yaml=MINI_SEEDS,

        persona_yaml=MINI_PERSONAS,

    )



    from testing.simulator import persona as persona_mod



    monkeypatch.setattr(

        persona_mod.PersonaEngine,

        "generate_opening",

        lambda self: f"Opening {self.persona.persona_id} {self.scenario.entity.get('order_id')}",

    )



    out = tmp_path / "out.jsonl"

    result = run_generation(

        output_path=out,

        max_limit=1,

        suite_path=suite_path,

        seed_override=None,

        persona_filters=["policy_prober"],

        randomize=False,

        stderr_print=lambda _m: None,

    )

    assert result.rows_written == 1

    row = json.loads(out.read_text(encoding="utf-8").strip())

    assert "policy_prober" in row["text"]

    assert re.search(r"ORD-\d{4}", row["text"])





def test_run_generation_raises_when_no_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    sim_root = tmp_path / "simulator"

    suite_path = _write_minimal_simulator_tree(

        sim_root,

        suite_yaml="""

run_id: mini

scenarios:

  - seed_id: s1

""",

        seed_yaml=MINI_SEEDS,

        persona_yaml=MINI_PERSONAS,

    )



    from testing.simulator import persona as persona_mod



    monkeypatch.setattr(

        persona_mod.PersonaEngine,

        "generate_opening",

        lambda self: (_ for _ in ()).throw(persona_mod.PersonaGenerationError("fail")),

    )



    out = tmp_path / "out.jsonl"

    with pytest.raises(RuntimeError, match="No rows were written"):

        run_generation(

            output_path=out,

            max_limit=2,

            suite_path=suite_path,

            seed_override=None,

            persona_filters=[],

            randomize=False,

            stderr_print=lambda _m: None,

            max_attempt_multiplier=3,

        )





def test_build_parser_defaults() -> None:

    parser = _build_parser()

    args = parser.parse_args(["--max-limit", "5"])

    assert args.max_limit == 5

    assert args.suite.endswith("regression.yaml") or "regression" in args.suite

    assert args.temperature is None





def test_build_parser_persona_nargs() -> None:

    parser = _build_parser()

    args = parser.parse_args(["--max-limit", "1", "--persona", "a", "b"])

    assert args.persona == ["a", "b"]





def test_build_parser_ollama_url_strips_trailing_slash() -> None:

    parser = _build_parser()

    args = parser.parse_args(["--max-limit", "1", "--ollama-url", "http://host:11434/"])

    assert args.ollama_url == "http://host:11434"





def test_build_parser_temperature_accepted() -> None:

    parser = _build_parser()

    args = parser.parse_args(["--max-limit", "1", "--temperature", "0.7"])

    assert args.temperature == pytest.approx(0.7)





def test_build_parser_temperature_rejects_out_of_range() -> None:

    parser = _build_parser()

    with pytest.raises(SystemExit):

        parser.parse_args(["--max-limit", "1", "--temperature", "2.01"])





def test_run_generation_temperature_override_propagates_to_persona_engine(

    tmp_path: Path, monkeypatch: pytest.MonkeyPatch

) -> None:

    sim_root = tmp_path / "simulator"

    suite_path = _write_minimal_simulator_tree(

        sim_root,

        suite_yaml="""

run_id: mini

scenarios:

  - seed_id: s1

""",

        seed_yaml=MINI_SEEDS,

        persona_yaml=MINI_PERSONAS,

    )



    from testing.simulator import persona as persona_mod



    captured: list[float | None] = []



    def fake_opening(self: persona_mod.PersonaEngine) -> str:

        captured.append(self.llm_temperature)

        return f"ok {self.scenario.entity.get('order_id')}"



    monkeypatch.setattr(persona_mod.PersonaEngine, "generate_opening", fake_opening)



    out = tmp_path / "out.jsonl"

    run_generation(

        output_path=out,

        max_limit=1,

        suite_path=suite_path,

        seed_override=None,

        persona_filters=[],

        randomize=False,

        temperature_override=1.25,

        stderr_print=lambda _m: None,

    )



    assert len(captured) == 1

    assert captured[0] == pytest.approx(1.25)





def test_count_jsonl_rows_skips_blank_lines(tmp_path: Path) -> None:

    path = tmp_path / "data.jsonl"

    path.write_text(

        '{"text": "a", "category": "order", "intent": "x", "seed_id": "s1", "persona": "p"}\n\n'

        '{"text": "b", "category": "refund", "intent": "y", "seed_id": "s2", "persona": "p"}\n',

        encoding="utf-8",

    )

    assert _count_jsonl_rows(path) == 2





def test_run_generation_resume_appends_to_existing_file(

    tmp_path: Path, monkeypatch: pytest.MonkeyPatch

) -> None:

    sim_root = tmp_path / "simulator"

    suite_path = _write_minimal_simulator_tree(

        sim_root,

        suite_yaml="""

run_id: mini

scenarios:

  - seed_id: s1

    persona_id: polite_first_timer

  - seed_id: s2

    persona_id: policy_prober

""",

        seed_yaml=MINI_SEEDS,

        persona_yaml=MINI_PERSONAS,

    )



    from testing.simulator import persona as persona_mod



    counter = {"n": 0}



    def fake_opening(self: persona_mod.PersonaEngine) -> str:

        counter["n"] += 1

        return f"Resume msg {counter['n']} {self.scenario.entity.get('order_id')}"



    monkeypatch.setattr(persona_mod.PersonaEngine, "generate_opening", fake_opening)



    out = tmp_path / "out.jsonl"

    out.write_text(

        json.dumps(

            {

                "text": "existing",

                "category": "order",

                "intent": "cancel_order",

                "seed_id": "s1",

                "persona": "polite_first_timer",

            }

        )

        + "\n",

        encoding="utf-8",

    )



    result = run_generation(

        output_path=out,

        max_limit=3,

        suite_path=suite_path,

        seed_override=None,

        persona_filters=[],

        randomize=False,

        existing_rows=1,

        stderr_print=lambda _m: None,

    )



    assert result.existing_rows == 1

    assert result.rows_written == 2

    assert result.total_rows == 3

    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(lines) == 3

    assert lines[0]["text"] == "existing"

    assert all("Resume msg" in row["text"] for row in lines[1:])





def test_run_generation_resume_noop_when_already_at_limit(tmp_path: Path) -> None:

    out = tmp_path / "out.jsonl"

    out.write_text(

        json.dumps(

            {

                "text": "existing",

                "category": "order",

                "intent": "cancel_order",

                "seed_id": "s1",

                "persona": "polite_first_timer",

            }

        )

        + "\n",

        encoding="utf-8",

    )



    result = run_generation(

        output_path=out,

        max_limit=1,

        suite_path=tmp_path / "missing.yaml",

        seed_override=None,

        persona_filters=[],

        randomize=False,

        existing_rows=1,

        stderr_print=lambda _m: None,

    )



    assert result.rows_written == 0

    assert result.total_rows == 1

    assert len(out.read_text(encoding="utf-8").splitlines()) == 1


