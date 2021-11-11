!!nginx Letencrypt!!
init:
	git clone --recurse-submodules https://github.com/evertramos/nginx-proxy-automation.git proxy 

start:
	cd proxy/bin && ./fresh-start.sh --yes -e thiti180536@gmail --skip-docker-image-check

start-api:
	docker-compose -f docker-compose-proxy.yml up -d

env:
	cd line-api && cp .env.sample .env