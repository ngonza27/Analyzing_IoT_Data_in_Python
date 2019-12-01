FROM node:latest

RUN mkdir /src

COPY ./web-server/ /src/

WORKDIR /src

RUN rm -rf node_modules

RUN ls

RUN npm install

EXPOSE 3000

CMD node server/server.js