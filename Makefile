.PHONY: docker-compose-dev-up
docker-compose-dev-up:
	docker compose -f deployments/docker/docker-compose.yml up --build -d

.PHONY: docker-compose-dev-down
docker-compose-dev-down:
	docker compose -f deployments/docker/docker-compose.yml down