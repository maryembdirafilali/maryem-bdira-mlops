from prefect import serve

from pipeline_prefect import all_flow, train_flow, evaluate_flow, code_flow

if __name__ == "__main__":
    serve(
        # pipeline complet : une fois par jour a 02:00
        all_flow.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",
            tags=["full-pipeline", "mlops"],
        ),
        # entrainement : une fois par jour a 03:00
        train_flow.to_deployment(
            name="ml-pipeline-train",
            cron="0 3 * * *",
            tags=["training", "mlops"],
        ),
        # lancement manuel uniquement
        evaluate_flow.to_deployment(
            name="ml-pipeline-evaluate",
            tags=["evaluation", "mlops"],
        ),
        code_flow.to_deployment(
            name="ml-pipeline-code",
            tags=["code", "mlops"],
        ),
    )
