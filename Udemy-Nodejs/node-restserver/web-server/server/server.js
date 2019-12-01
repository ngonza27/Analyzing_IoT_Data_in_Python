require('./config/config');

const express = require('express');
const mongoose = require('mongoose');
const path = require('path');
const app = express();

const bodyParser = require('body-parser');

// Important commands:
// run the app -> $node | nodemon server.js
// run db -> $sudo mongod

// midlewear
// parse application/x-www-form
app.use(bodyParser.urlencoded({extended: false}));
// parse application/json
app.use(bodyParser.json());

// habilitar carpeta public
app.use( express.static(path.resolve(__dirname,'../public')));

// configuracion global de rutas
app.use(require('./routes/index') );


mongoose.connect('mongodb://3.212.173.121:27017/comentarios', (err, res) =>{
    if(err) throw err;
    console.log('Base de datos online');
});


app.listen(process.env.PORT, () => {
    console.log('Escuchando puerto: ', process.env.PORT);
})