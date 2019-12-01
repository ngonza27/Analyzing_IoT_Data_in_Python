const mongoose = require('mongoose');
const uniqueValidator = require('mongoose-unique-validator');

let rolesValidos = {
    values: ['ADMIN_ROLE', 'USER_ROLE'],
    message: '{VALUE} no es un rol valido'
};

let Schema = mongoose.Schema;

let usuarioSchema = new Schema({
    nombre: {
        type: String,
        require: [true, 'El nombre es necesario']
    },
    edad: {
        type: String,
        required: [true, 'La edad es necesaria']
    },
    ciudad: {
        type: String,
        required: [true, 'Su ciudad de residencia es necesaria']
    },
    direccion: {
        type: String,
        required: [true, 'Su direccion es necesaria']
    },
    estrato: {
        type: String,
        required: [true, 'Su estrato es necesario']
    },
    email: {
        type: String,
        unique: true,
        required: [true, 'Por favor ingrese un email']
    },
    password: {
        type: String,
        required: [true, 'Porfavor ingrese una contraseña']
    },
    img: {
        type: String,
        required: false
    },
    role: {
        type: String,
        default: 'USER_ROLE',
        enum: rolesValidos
    },
    estado: {
        type: Boolean,
        default: true
    },
    google: {
        type: Boolean,
        default: false
    }
});

usuarioSchema.methods.toJSON = function() {
    let user = this;
    let userObject = user.toObject();
    delete userObject.password;

    return userObject;
}  

usuarioSchema.plugin(uniqueValidator, {message: '{PATH} debe de ser unico'} )

module.exports = mongoose.model('Usuario', usuarioSchema);