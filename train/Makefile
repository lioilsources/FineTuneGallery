.PHONY: build train shell

build:
	docker build -t finetune-kohya:latest .

train:
	@test -n "$(DATASET)" || (echo "usage: make train DATASET=<name>"; exit 1)
	./run.sh $(DATASET)

shell:
	docker run --rm -it --gpus all --entrypoint bash finetune-kohya:latest
