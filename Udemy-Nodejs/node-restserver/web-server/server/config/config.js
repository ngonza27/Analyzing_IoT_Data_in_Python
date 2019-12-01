



// ================
// Puerto
// ================

process.env.PORT = process.env.PORT || 3000;


// ================
// Vencimiento del token
// ================
// 60 segundos
// 60 minutos
// 24 horas
// 30 dias
process.env.CADUCIDAD_TOKEN = '48h';


// ================
// SEED de autenticacion
// ================
process.env.SEED = process.env.SEED || 'este-es-el-seed-desarrollo';



// ================
// Google client id
// ================

process.env.CLIENT_ID = process.env.CLIENT_ID || '527836340157-34r6m2ipma3hpcqne8jq4dudkjdmv5ol.apps.googleusercontent.com';