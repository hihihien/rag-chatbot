# frontend/Dockerfile

FROM node:18-alpine

WORKDIR /app

COPY . /app

RUN npm install

EXPOSE 3000

CMD ["npm", "run", "dev"]
