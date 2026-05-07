.PHONY: train build run mlflow-ui deploy undeploy

train:
	python src/train.py

build:
	docker build -t churn-api:latest .

run:
	docker run --rm -p 8000:8000 churn-api:latest

mlflow-ui:
	mlflow ui --port 5000

deploy:
	kubectl apply -f k8s/

undeploy:
	kubectl delete -f k8s/
