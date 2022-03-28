#version 410

out vec4 frag_colour;
uniform vec4 colour = vec4 (1.0, 0.0, 0.0, 1.0);
void main() {
	frag_colour = colour;
} 
