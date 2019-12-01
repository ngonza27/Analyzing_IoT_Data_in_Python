const mongoose = require('mongoose')
const Schema = mongoose.Schema;

let categoriaSchema = new Schema({
    nombre: { type: String, unique: true},
    descripcion: {type: String, required: [true, 'La descripción es obligatoria'] },
    usuario: { type: Schema.Types.ObjectId, ref: 'Usuario' }
});


module.exports = mongoose.model('Categoria', categoriaSchema);