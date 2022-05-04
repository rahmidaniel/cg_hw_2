//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

int frame = 0;

struct Quaternion{
    float w;
    vec3 xyz;
    Quaternion(vec3 axis, float angle) {
        xyz = axis * sin(angle / 2);
        w = cos(angle / 2);
    }
    Quaternion inv(){
        Quaternion q1(xyz, w);
        q1.w *= -1;
        return q1;
    }
};
Quaternion mul(Quaternion q1, Quaternion q2){
    return {q1.w * q2.xyz + q2.w * q1.xyz + cross(q1.xyz, q2.xyz),q1.w * q2.w - dot(q1.xyz, q2.xyz)};
}

vec3 quatRot(Quaternion q1, vec3 p) {
    Quaternion qInv = q1.inv();
    return mul(mul(q1,{p, 0}),qInv).xyz;
}

Quaternion Rotate(vec3 u, vec3 v){
    vec3 h = u + v / length(u+v);
    vec3 c = cross(u,h);
    return {c, dot(u,h)};
}

struct Material {
    vec3 ka, kd, ks;
    float  shininess;
    Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd * M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
    float t;
    vec3 position, normal;
    Material * material;
    Hit() { t = -1; }
};

struct Ray {
    vec3 start, dir;
    Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
    Material * material;
public:
    virtual Hit intersect(const Ray& ray) = 0;
};

float combine(float t1, float t2, vec3 normal1, vec3 normal2, vec3& normal) {
    if (t1 < 0.0 && t2 < 0.0) {
        return -1.0;
    } else if (t2 < 0.0) {
        normal = normal1;
        return t1;
    } else if (t1 < 0.0) {
        normal = normal2;
        return t2;
    } else {
        if (t1 < t2) {
            normal = normal1;
            return t1;
        } else {
            normal = normal2;
            return t2;
        }
    }
}

struct Sphere : public Intersectable {
    vec3* center;
    float radius;

    Sphere(vec3* _center, float _radius, Material* _material) {
        center = _center;
        radius = _radius;
        material = _material;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;
        vec3 dist = ray.start - *center;
        float a = dot(ray.dir, ray.dir);
        float b = dot(dist, ray.dir) * 2.0f;
        float c = dot(dist, dist) - radius * radius;
        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
        float t2 = (-b - sqrt_discr) / 2.0f / a;

        if (t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;

        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = normalize(hit.position - *center);
        hit.material = material;
        return hit;
    }
};

struct Paraboloid : public Intersectable {
    vec3* center;
    float radius;

    Paraboloid(vec3* _center, float _radius, Material* _material) {
        center = _center;
        radius = _radius;
        material = _material;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;
        vec3 dist = ray.start - *center;

        vec2 rayXZ {ray.dir.x, ray.dir.z};
        vec2 distXZ {dist.x, dist.z};

        float a = dot(rayXZ, rayXZ);
        float b = dot(distXZ, rayXZ) * 2.0f - ray.dir.y; // * 0.5
        float c = dot(distXZ, distXZ) - dist.y; // * 0.5
        float discr = b * b - 4.0f * a * c;

        if (discr < 0.f) return hit;

        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / (2.f * a);	// t1 >= t2 for sure
        float t2 = (-b - sqrt_discr) / (2.f * a);

        vec3 hitPos1 = ray.start + ray.dir * t1;
        vec3 hitPos2 = ray.start + ray.dir * t2;

        hit.t = combine(t1, t2, hitPos1, hitPos2, hit.position);
        if (length(hit.position - *center) > radius) hit.t = -1.0;

        vec3 dx = vec3(1,2 * hit.position.x,0);
        vec3 dz = vec3(0,2 * hit.position.z,1);

        hit.normal = normalize(cross(dx, dz));
        hit.material = material;
        return hit;
    }
};

struct Cylinder : public Intersectable {
    vec3* center;
    float radius, height;

    Cylinder(vec3* _center, float _radius, float _height, Material* _material) {
        center = _center;
        radius = _radius;
        height = _height;
        material = _material;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;
        vec3 dist = ray.start - *center;

        vec2 rayXZ {ray.dir.x, ray.dir.z};
        vec2 distXZ {dist.x, dist.z};

        float a = dot(rayXZ, rayXZ);
        float b = dot(distXZ, rayXZ) * 2.0f;
        float c = dot(distXZ, distXZ) - radius * radius;
        float discr = b * b - 4.0f * a * c;

        if (discr < 0.f) return hit;

        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / (2.f * a);	// t1 >= t2 for sure
        float t2 = (-b - sqrt_discr) / (2.f * a);

        vec3 hitPos1 = ray.start + ray.dir * t1;
        vec3 hitPos2 = ray.start + ray.dir * t2;
        if (hitPos1.y < center->y || hitPos1.y > height + center->y) t1 = -1.0;
        if (hitPos2.y < center->y || hitPos2.y > height + center->y) t2 = -1.0;

        hit.t = combine(t1, t2, hitPos1, hitPos2, hit.position);

        hit.normal = hit.position - *center;
        hit.normal.y = 0.f;
        hit.material = material;
        return hit;
    }
};

struct Plane : public Intersectable{
    vec3 point, normal;
    float radius;
    Plane(const vec3& _point, float _radius, const vec3& _normal, Material* _material){
        point = _point;
        radius = _radius;
        normal = _normal;
        material = _material;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;
        float t = dot(point - ray.start, normal) / dot(ray.dir, normal);
        if(t < 0.f) return hit;

        hit.position = ray.start + ray.dir * t;
        if(radius != 0 && hit.position.x * hit.position.x + hit.position.z * hit.position.z > radius * radius) t = -1.f;

        hit.t = t;
        hit.normal = normal;
        hit.material = material;
        return hit;
    }
};

class Camera {
    vec3 eye, lookat, right, up;
    float fov;
public:
    void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
        eye = _eye;
        lookat = _lookat;
        fov = _fov;
        vec3 w = eye - lookat;
        float focus = length(w);
        right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
        up = normalize(cross(w, right)) * focus * tanf(fov / 2);
    }
    Ray getRay(int X, int Y) {
        vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
        return Ray(eye, dir);
    }

    void animate(float dt){
        vec3 d = eye - lookat;
        //Quaternion q(normalize(vec3(1,6,3)), frame/60.f);
        //eye = quatRot(q, lookat);
        eye = vec3(d.x * cos(dt) + d.z * sin(dt), d.y, -d.x * sin(dt) + d.z * cos(dt)) + lookat;
        set(eye, lookat, up, fov);
    }
};

struct Light {
    vec3 direction;
    vec3 Le;
    Light(vec3 _direction, vec3 _Le) {
        direction = normalize(_direction);
        Le = _Le;
    }
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

class Scene {
    std::vector<Intersectable *> objects;
    std::vector<Light *> lights;
    Camera camera;
    vec3 La;

    std::vector<vec3*> nodes;
public:
    void build() {
        vec3 eye = vec3(0, 0, 2), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
        float fov = 45 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        La = vec3(0.05f, 0.05f, 0.05f);
        vec3 lightDirection(1, 2, 3), Le(2, 2, 2);
        lights.push_back(new Light(lightDirection, Le));

        vec3 kd(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
        vec3 kd2(0.3f, 0.2f, 0.3f), ks2(1, 1, 1);
        auto * material = new Material(kd, ks, 40);
        auto * material2 = new Material(kd2, ks2, 30);

        float deskY = -0.4;
        float jointR = 0.02;
        float baseR = 0.1, baseH = 0.03;
        float neckR = 0.015, neckH = 0.25;

        /// vectors
            nodes.emplace_back(new vec3(0,deskY,0)); // dont move
            //nodes.emplace_back(new vec3(0,deskY + baseH,0)); // dont move

            // child 1
            //nodes.emplace_back(new vec3(0,deskY + baseH,0));
            nodes.emplace_back(new vec3(0,deskY + baseH,0)); // dont move

            nodes.emplace_back(new vec3(0,deskY + baseH,0));

            // child 2
            //nodes.emplace_back(new vec3(0,deskY + baseH + neckH,0));
            nodes.emplace_back(new vec3(0,deskY + baseH + neckH,0));

            // child 3
            //nodes.emplace_back(new vec3(0,deskY + baseH + neckH * 2,0));
            nodes.emplace_back(new vec3(0,deskY + baseH + neckH * 2,0));
        ///
        //0
        objects.push_back(new Plane(vec3(0,deskY,0), 0, vec3(0,1,0), material2));
        //1,2
        objects.push_back(new Cylinder(nodes[0],baseR, baseH, material));
        objects.push_back(new Plane(*nodes[1], baseR, vec3(0,1,0), material));
        //3,4
        objects.push_back(new Sphere(nodes[2], jointR, material));
        objects.push_back(new Cylinder(nodes[2],neckR, neckH, material));
        //5,6
        objects.push_back(new Sphere(nodes[3], jointR, material));
        objects.push_back(new Cylinder(nodes[3],neckR,neckH, material));
        //7,8
        objects.push_back(new Sphere(nodes[4], jointR, material));
        objects.push_back(new Paraboloid(nodes[4], 0.3, material));

        // light
        //lights.push_back(new Light(vec3(0,deskY + baseH + neckH * 2,0.2), Le * 0.5f));
    }

    void render(std::vector<vec4>& image) {
        for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
            for (int X = 0; X < windowWidth; X++) {
                vec3 color = trace(camera.getRay(X, Y));
                image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
            }
        }
    }

    void animate(float dt){
        //camera.animate(dt);
        // do rotations
        //*nodes[4] = quatRot(Quaternion(normalize(vec3(1,2,3)), sin(frame/60.f)), *nodes[4]);
//        mat4 rot = RotationMatrix(frame/60.f, vec3(6,6,3));
//        vec3 a = *nodes[3];
//        vec4 q = vec4(a.x, a.y, a.z, 1) * rot;
//        *nodes[3] = vec3(q.x, q.y, q.z);
//
//        *nodes[2] = quatRot({normalize(vec3(1,6,3)), ((float)frame)/60.f}, *nodes[2]);
    }

    Hit firstIntersect(Ray ray) {
        Hit bestHit;
        for(int i=0; i < objects.size(); i++) {
            Ray r(ray.start, ray.dir);
            if(i > 2 && i < 5){
                Quaternion q(normalize(vec3(1,2,3)), sin((float)frame/60.f));
                r.start = quatRot(q, r.start);
                r.dir = quatRot(q, r.dir);

                Hit hit = objects[i]->intersect(r); //  hit.t < 0 if no intersection
                //hit.normal = quatRot(q.inv(), hit.normal);

                if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
            } else if(i > 4 && i < 7){
                Hit hit = objects[i]->intersect(r); //  hit.t < 0 if no intersection
                if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
            } else {
                Hit hit = objects[i]->intersect(r); //  hit.t < 0 if no intersection
                if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
            }
        }
//        if(objects.size() < 2) return bestHit;
//        Hit hit1 = objects[0]->intersect(ray);
//        Hit hit2 = objects[1]->intersect(ray);
//        vec3 normal;
//        float t = combine(hit2.t, hit1.t, hit2.position, hit1.position, normal);
//        for(int i=1; i < objects.size(); i++) {
//            Hit hit = objects[i]->intersect(ray);
//            t = combine(t, hit.t, normal, hit.position, normal);
//        }
//        bestHit.t = t;
//        bestHit.position = normal;

        if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
        return bestHit;
    }

    bool shadowIntersect(Ray ray) {	// for directional lights
        for (Intersectable * object : objects) if (object->intersect(ray).t > 0) return true;
        return false;
    }

    vec3 trace(Ray ray, int depth = 0) {
        Hit hit = firstIntersect(ray);
        if (hit.t < 0) return La;
        vec3 outRadiance = hit.material->ka * La;
        for (Light * light : lights) {
            Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
            float cosTheta = dot(hit.normal, light->direction);
            if (cosTheta > 0.f && !shadowIntersect(shadowRay)) {	// shadow computation
                outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
                vec3 halfway = normalize(-ray.dir + light->direction);
                float cosDelta = dot(hit.normal, halfway);
                if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
            }
        }
        return outRadiance;
    }
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";

class FullScreenTexturedQuad {
    unsigned int vao;	// vertex array object id and texture id
    Texture texture;
public:
    FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
            : texture(windowWidth, windowHeight, image)
    {
        glGenVertexArrays(1, &vao);	// create 1 vertex array object
        glBindVertexArray(vao);		// make it active

        unsigned int vbo;		// vertex buffer objects
        glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

        // vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
    }

    void Draw() {
        glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
        gpuProgram.setUniform(texture, "textureUnit");
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
    }

    void LoadTexture(std::vector<vec4> &image) {
        glBindTexture(GL_TEXTURE_2D, texture.textureId);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
    }
};

FullScreenTexturedQuad * fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();

    std::vector<vec4> image(windowWidth * windowHeight);
    long timeStart = glutGet(GLUT_ELAPSED_TIME);
    scene.render(image);
    long timeEnd = glutGet(GLUT_ELAPSED_TIME);
    printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

    // copy image to GPU as a texture
    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

    // create program for the GPU
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
    std::vector<vec4> image(windowWidth * windowHeight);
    frame++;
    scene.render(image);
    fullScreenTexturedQuad->LoadTexture(image);
    fullScreenTexturedQuad->Draw();
    glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    scene.animate(0.05f);
    glutPostRedisplay();
}

