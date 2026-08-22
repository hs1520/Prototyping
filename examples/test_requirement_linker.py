"""
Quick smoke-test: run requirement_linker against the AutonomousDrone model
and print the generated .parm file + coverage report.
"""

from src.prototyping.provider_factory import create_llm
from src.app.pipeline import PrototypingPipeline
from src.sitl.requirement_linker import RequirementLinker

from drone_system_v2 import DRONE_DESCRIPTION, DRONE_REQUIREMENTS


def main():
    import pathlib
    import pickle
    cache = pathlib.Path("/tmp/drone_model_cache.pkl")
    if cache.exists():
        print("[cache hit] loading model from /tmp/drone_model_cache.pkl")
        with open(cache, "rb") as f:
            model = pickle.load(f)
    else:
        llm = create_llm(provider="vertex")
        pipeline = PrototypingPipeline(llm=llm, quality_threshold=0.75,
                                       max_iterations=1, verbose=False)
        result = pipeline.generate_system(
            system_name="AutonomousDrone",
            description=DRONE_DESCRIPTION,
            additional_requirements=DRONE_REQUIREMENTS,
        )
        model = result["model"]
        with open(cache, "wb") as f:
            pickle.dump(model, f)
        print(f"[cache saved] → {cache}")
    linker = RequirementLinker(model)

    print("=" * 60)
    print("COVERAGE REPORT")
    print("=" * 60)
    print(linker.coverage_report())

    print()
    print("=" * 60)
    print("GENERATED .parm FILE")
    print("=" * 60)
    print(linker.generate_parm_file())

    print("=" * 60)
    print("L2 TEST SPECS")
    print("=" * 60)
    for spec in linker.generate_test_specs():
        if spec.tier == "L2":
            print(f"[{spec.req_id}]")
            print(f"  inject : {spec.inject}")
            print(f"  verify : {spec.verify}")
            print(f"  notes  : {spec.notes}")
            print()


if __name__ == "__main__":
    main()
