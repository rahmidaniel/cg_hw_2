#version 330 
in vec4 vertexPosition; 
out vec2 texCoord; 

void main() { 
    gl_Position = vertexPosition; 
    texCoord = vertexPosition.xy * 0.5 + 0.5; 
}