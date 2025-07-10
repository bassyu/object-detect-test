FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt install -y curl
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash
RUN apt install -y nodejs

WORKDIR /app
COPY . .
RUN npm install

CMD ["npm", "run", "start"]
