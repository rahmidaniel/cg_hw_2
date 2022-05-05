#version 330 
in vec4 vertexPosition;
uniform vec3 wLookAt, wRight, wUp;          // pos of eye

layout(location = 0) in vec2 cCamWindowVertex;	// Attrib Array 0
out vec3 p;

void main() {
    gl_Position = vec4(cCamWindowVertex, 0, 1);
    p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
}