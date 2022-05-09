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
// Nev    : Daniel Rahmi
// Neptun : WIFRTR
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

/**
 * A program alapjait az eloadasok anyagabol, a raytracing gpu pelda programbol es a grafika konzultaciobol alkottam meg.
 * */
#include "framework.h"

int frame = 0;

struct Light {
    vec3 position;
    vec3 Le, La;

    Light(const vec3 &position = 0, const vec3 &le = 0, const vec3 &la=0) : position(position), Le(le), La(la) {}
};

struct Sphere {
    vec3 center;
    float radius;

    Sphere(const vec3 &center=0, float radius=0) : center(center), radius(radius) {}
};
struct Paraboloid {
    vec3 center, dir;
    float radius;

    Paraboloid(const vec3 &center=0, const vec3 &dir=0, float radius=0) : center(center), dir(dir), radius(radius) {}
};
struct Cylinder {
    vec3 center, dir;
    float radius, height;

    Cylinder(const vec3 &center=0, const vec3 &dir=0, float radius=0, float height=0) : center(center), dir(dir),
                                                                                radius(radius), height(height) {}
};
struct Plane {
    vec3 center, normal;
    float radius;

    Plane(const vec3 &center=0, const vec3 &normal=0, float radius=0) : center(center), normal(normal), radius(radius) {}
};

vec4 quat(vec3 axis, float angle) {
    vec3 a = axis * sin(angle / 2);
    return vec4(a.x,a.y,a.z, cos(angle / 2));
}

vec4 quatInv(vec4 q1) {
    return vec4(-q1.x,-q1.y,-q1.z, q1.w);
}

vec4 quatMul(vec4 q1, vec4 q2) {
    vec3 q1r = vec3(q1.x,q1.y,q1.z);
    vec3 q2r = vec3(q2.x,q2.y,q2.z);
    vec3 a = q1.w * q2r + q2.w * q1r + cross(q1r, q2r);
    float b = q1.w * q2.w - dot(q1r, q2r);
    return vec4(a.x,a.y,a.z,b);
}

vec3 quatRot(vec4 q1, vec3 p) {
    vec4 qInv = quatInv(q1);
    vec4 a = quatMul(quatMul(q1, vec4(p.x,p.y,p.z, 0)), qInv);
    return vec3(a.x,a.y,a.z);
}

Cylinder rotate(Cylinder object, vec4 q, vec3 center){
    return Cylinder(quatRot(q, object.center - center) + center, quatRot(q, object.dir), object.radius, object.height);
}

Paraboloid rotate(Paraboloid object, vec4 q, vec3 center){
    return Paraboloid(quatRot(q, object.center - center) + center, quatRot(q, object.dir), object.radius);
}

Sphere rotate(Sphere object, vec4 q, vec3 center){
    return Sphere(quatRot(q, object.center - center) + center, object.radius);
}

Light rotate(Light object, vec4 q, vec3 center){
    return Light(quatRot(q, object.position - center) + center, object.Le, object.La);
}

Plane plane;
Sphere joints[3];
Cylinder base; Plane baseTop;
Cylinder arms[2];
Paraboloid lamp;
Light lightBulb;

void animates(float dt){
    float time = dt;
    vec4 q1 = quat(normalize(vec3(1,5,3)), time);
    vec4 q2 = quat(normalize(vec3(3,3,3)), time);
    vec4 q3 = quat(normalize(vec3(1,1,0)), time);

    arms[0] = rotate(arms[0], q1, joints[0].center);
    joints[1] = rotate(joints[1], q1, joints[0].center);
    arms[1] = rotate(arms[1], q1, joints[0].center);
    joints[2] = rotate(joints[2], q1, joints[0].center);
    lamp = rotate(lamp, q1, joints[0].center);
    lightBulb = rotate(lightBulb, q1, joints[0].center);

    arms[1] = rotate(arms[1], q2, joints[1].center);
    joints[2] = rotate(joints[2], q2, joints[1].center);
    lamp = rotate(lamp, q2, joints[1].center);
    lightBulb = rotate(lightBulb, q2, joints[1].center);

    lamp = rotate(lamp, q3, joints[2].center);
    lightBulb = rotate(lightBulb, q3, joints[2].center);
}

void builds(){
    float deskY = -4;
    float jointR = 0.2;
    float baseR = 1.5, baseH = 0.3;
    float neckR = 0.15, neckH = 2.5;

    vec3 planeLevel = vec3(0,deskY,0);
    plane = Plane(planeLevel, vec3(0,1,0), 0);
    base = Cylinder(planeLevel, planeLevel, baseR, baseH);
    baseTop = Plane(planeLevel + vec3(0,baseH,0), vec3(0,1,0), baseR);

    joints[0] = Sphere(baseTop.center, jointR);
    arms[0] = Cylinder(baseTop.center, vec3(0,1,0), neckR, neckH);
    joints[1] = Sphere(arms[0].center + vec3(0, neckH, 0), jointR);
    arms[1] = Cylinder(joints[1].center, vec3(0,1,0), neckR, neckH);
    joints[2] = Sphere(arms[1].center + vec3(0, neckH, 0), jointR);
    lamp = Paraboloid(joints[2].center + vec3(0,joints[2].radius,0), vec3(0,1,0), 3);

    lightBulb = Light(lamp.center + lamp.dir * 1.5f, vec3(200,200,200), vec3(0,0,0));
}

struct Material {
    vec3 ka, kd, ks;
    float  shininess;
    Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd * M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

class Camera {
public:
    vec3 eye, lookat, right, up;
    float fov;

    void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
        eye = _eye;
        lookat = _lookat;
        fov = _fov;
        vec3 w = eye - lookat;
        float focus = length(w);
        right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
        up = normalize(cross(w, right)) * focus * tanf(fov / 2);
    }

    void animate(float dt){
        eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x, eye.y,-(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
        set(eye, lookat, up, fov);
    }
};

//---------------------------
class Shader : public GPUProgram {
//---------------------------
public:
    void setUniformMaterials(const std::vector<Material*>& materials) {
        char name[256];
        for (unsigned int mat = 0; mat < materials.size(); mat++) {
            sprintf(name, "materials[%d].ka", mat); setUniform(materials[mat]->ka, name);
            sprintf(name, "materials[%d].kd", mat); setUniform(materials[mat]->kd, name);
            sprintf(name, "materials[%d].ks", mat); setUniform(materials[mat]->ks, name);
            sprintf(name, "materials[%d].shininess", mat); setUniform(materials[mat]->shininess, name);
        }
    }

    void setUniformCamera(const Camera& camera) {
        setUniform(camera.eye, "wEye");
        setUniform(camera.lookat, "wLookAt");
        setUniform(camera.right, "wRight");
        setUniform(camera.up, "wUp");
    }

    void setUniformObjects() {
        char name[256];
        setUniform(plane.center, "plane.center");
        setUniform(plane.normal, "plane.normal");
        setUniform(plane.radius, "plane.radius");

        for (int i = 0; i < 3; ++i) {
            sprintf(name, "joints[%d].center", i);  setUniform(joints[i].center, name);
            sprintf(name, "joints[%d].radius", i);  setUniform(joints[i].radius, name);
        }

        for (int i = 0; i < 2; ++i) {
            sprintf(name, "arms[%d].center", i);  setUniform(arms[i].center, name);
            sprintf(name, "arms[%d].radius", i);  setUniform(arms[i].radius, name);
            sprintf(name, "arms[%d].height", i);  setUniform(arms[i].height, name);
            sprintf(name, "arms[%d].dir", i);  setUniform(arms[i].dir, name);
        }

        setUniform(base.center, "base.center");
        setUniform(base.radius, "base.radius");
        setUniform(base.height, "base.height");
        setUniform(base.dir, "base.dir");

        setUniform(baseTop.center, "baseTop.center");
        setUniform(baseTop.normal, "baseTop.normal");
        setUniform(baseTop.radius, "baseTop.radius");

        setUniform(lamp.center, "lamp.center");
        setUniform(lamp.dir, "lamp.dir");
        setUniform(lamp.radius, "lamp.radius");

        setUniform(lightBulb.La, "lightBulb.La");
        setUniform(lightBulb.Le, "lightBulb.Le");
        setUniform(lightBulb.position, "lightBulb.position");

    }
};

class Scene {
    std::vector<Material *> materials;
    Camera camera;

    std::vector<vec3*> nodes;
public:
    void build() {
        vec3 eye = vec3(0, 0, 50), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
        float fov = 45 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        vec3 kd(0.1f, 0.2f, 0.5f), ks(10, 10, 10);
        vec3 kd2(0.4f, 0.2f, 0.1f), ks2(10, 10, 10);
        materials.push_back(new Material(kd, ks, 40));
        materials.push_back(new Material(kd2, ks2, 400));

        builds();
    }
    void setUniform(Shader& shader) {
        shader.setUniformMaterials(materials);
        shader.setUniformCamera(camera);
        shader.setUniformObjects();
    }

    void animate(float dt){
        animates(dt);
        camera.animate(dt/2.f);
    }
};

Scene scene;
Shader shader;

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	uniform vec3 wLookAt, wRight, wUp;          // pos of eye

	layout(location = 0) in vec2 cCamWindowVertex;	// Attrib Array 0
	out vec3 p;

	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(#version 330 core

const float PI = 3.141592654;
const int parts = 9;

struct Material {
    vec3 ka, kd, ks;
    float  shininess;
};

struct Light {
    vec3 position;
    vec3 Le, La;
};

struct Sphere {
    vec3 center;
    float radius;
};
struct Paraboloid {
    vec3 center, dir;
    float radius;
};
struct Cylinder {
    vec3 center, dir;
    float radius, height;
};
struct Plane {
    vec3 center, normal;
    float radius;
};

struct Hit {
    float t;
    vec3 position, normal;
    int mat;	// material index
};

struct Ray {
    vec3 start, dir;
};

struct Shape{
    vec3 center;
    float radius;
};

uniform vec3 wEye;
uniform Light light;
uniform Material materials[2];  // diffuse, specular, ambient ref
uniform int frame;

in  vec3 p;					// center on camera window corresponding to the pixel
out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

vec4 quat(vec3 axis, float angle) {
    return vec4(axis * sin(angle / 2), cos(angle / 2));
}

vec4 quatInv(vec4 q1) {
    return vec4(-q1.xyz, q1.w);
}

vec4 quatMul(vec4 q1, vec4 q2) {
    return vec4(q1.w * q2.xyz + q2.w * q1.xyz + cross(q1.xyz, q2.xyz),q1.w * q2.w - dot(q1.xyz, q2.xyz));
}

vec3 quatRot(vec4 q1, vec3 p) {
    vec4 qInv = quatInv(q1);
    return quatMul(quatMul(q1, vec4(p, 0)), qInv).xyz;
}

Cylinder rotate(Cylinder object, vec4 q, vec3 center){
    return Cylinder(quatRot(q, object.center - center) + center, quatRot(q, object.dir), object.radius, object.height);
}

Paraboloid rotate(Paraboloid object, vec4 q, vec3 center){
    return Paraboloid(quatRot(q, object.center - center) + center, quatRot(q, object.dir), object.radius);
}

Sphere rotate(Sphere object, vec4 q, vec3 center){
    return Sphere(quatRot(q, object.center - center) + center, object.radius);
}

Light rotate(Light object, vec4 q, vec3 center){
    return Light(quatRot(q, object.position - center) + center, object.Le, object.La);
}

float combine(float t1, float t2, vec3 normal1, vec3 normal2, out vec3 normal) {
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

Hit intersect(const Sphere obj, const Ray ray) {
    Hit hit;
    hit.mat = 0;
    hit.t = -1.f;
    vec3 oc = ray.start - obj.center;
    float a = dot(ray.dir, ray.dir);
    float b = 2.0 * dot(oc, ray.dir);
    float c = dot(oc, oc) - obj.radius * obj.radius;
    float disc = b * b - 4 * a * c;
    if (disc < 0.0) return hit;

    hit.t = (-b - sqrt(disc)) / (2 * a);
    hit.position = ray.start + ray.dir * hit.t;
    hit.normal = normalize(hit.position - obj.center);

    return hit;
}

Hit intersect(const Plane obj, const Ray ray) {
    Hit hit;

    hit.t = dot(obj.center - ray.start, obj.normal) / dot(ray.dir, obj.normal);
    hit.position = ray.start + ray.dir * hit.t;
    hit.mat = 0;
    if(obj.radius > 0 && hit.position.x * hit.position.x + hit.position.z * hit.position.z > obj.radius * obj.radius) hit.t= -1.0;
    else if(obj.radius == 0.0) hit.mat = 1;

    hit.normal = obj.normal;

    return hit;
}

Hit intersect(const Cylinder object, const Ray r) {
    Hit hit;
    hit.mat = 0;
    hit.t = -1;

    vec3 halfway = normalize(object.dir + vec3(0,1,0));
    vec4 q = quat(halfway, PI);
    Ray ray = Ray(quatRot(q, r.start), quatRot(q, r.dir));

    Cylinder obj = rotate(object, q, vec3(0,0,0));

    vec2 oc = ray.start.xz - obj.center.xz;
    float a = dot(ray.dir.xz, ray.dir.xz);
    float b = 2.0 * dot(oc, ray.dir.xz);
    float c = dot(oc, oc) - obj.radius * obj.radius;
    float disc = b * b - 4 * a * c;
    if (disc < 0.0) return hit;

    float t1 = (-b - sqrt(disc)) / (2 * a);
    float t2 = (-b + sqrt(disc)) / (2 * a);
    vec3 hitPos1 = ray.start + ray.dir * t1;
    vec3 hitPos2 = ray.start + ray.dir * t2;

    if (hitPos1.y < obj.center.y || hitPos1.y > obj.height + obj.center.y) t1 = -1.0;
    if (hitPos2.y < obj.center.y || hitPos2.y > obj.height + obj.center.y) t2 = -1.0;

    hit.t = combine(t1, t2, hitPos1, hitPos2, hit.position);

    hit.normal = hit.position - obj.center;
    hit.normal.y = 0.0;
    hit.normal = normalize(hit.normal);

    hit.position = quatRot(q, hit.position);
    hit.normal = quatRot(q, hit.normal);

    return hit;
}

Hit intersect(const Paraboloid object, const Ray r){
    Hit hit;
    hit.mat = 0;
    hit.t = -1;

    vec3 halfway = normalize(object.dir + vec3(0,1,0));
    vec4 q = quat(halfway, PI);
    Ray ray = Ray(quatRot(q, r.start), quatRot(q, r.dir));

    Paraboloid obj = rotate(object, q, vec3(0,0,0));

    vec3 oc = ray.start - obj.center;
    float a = dot(ray.dir.xz, ray.dir.xz);
    float b = 2 * dot(oc.xz, ray.dir.xz) - ray.dir.y;
    float c = dot(oc.xz, oc.xz) - oc.y;
    float disc = b * b - 4 * a * c;
    if (disc < 0.0) return hit;

    float t1 = (-b - sqrt(disc)) / (2 * a);
    float t2 = (-b + sqrt(disc)) / (2 * a);

    vec3 hitPos1 = ray.start + ray.dir * t1;
    vec3 hitPos2 = ray.start + ray.dir * t2;

    if (length(hitPos1- obj.center) > obj.radius) t1 = -1.0;
    if (length(hitPos2- obj.center) > obj.radius) t2 = -1.0;

    hit.t = combine(t1, t2, hitPos1, hitPos2, hit.position);

    hit.normal = quatRot(q, normalize(vec3((hit.position.x - obj.center.x) * 2, - 1, (hit.position.z - obj.center.z) * 2)));
    hit.position = quatRot(q, hit.position);

    return hit;
}

uniform Plane plane;
uniform Sphere joints[3];
uniform Cylinder base; uniform Plane baseTop;
uniform Cylinder arms[2];
uniform Paraboloid lamp;
uniform Light lightBulb;

Hit hits[parts];

void runRay(Ray ray, out Hit hits[parts]){
    int c = 0;
    hits[c++] = intersect(plane, ray);
    hits[c++] = intersect(base, ray);
    hits[c++] = intersect(baseTop, ray);

    hits[c++] = intersect(joints[0], ray);;
    hits[c++] = intersect(arms[0], ray);
    hits[c++] = intersect(joints[1], ray);
    hits[c++] = intersect(arms[1], ray);
    hits[c++] = intersect(joints[2], ray);
    hits[c++] = intersect(lamp, ray);
}

void checkHit(Hit hit, out Hit bestHit){
    if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) bestHit = hit;
}

Hit firstIntersect(Ray ray) {
    Hit bestHit;
    bestHit.t = -1;

    runRay(ray, hits);

    //checkHit(hits[i], bestHit);
    //float t = combine(hits[1].t, hits[0].t, hits[1].normal, hits[0].normal, bestHit.normal);
    for(int i = 0; i < parts; i++) {
        //t = combine(t, hits[i].t, bestHit.normal, hits[i].normal, bestHit.normal);
        checkHit(hits[i], bestHit);
    }
    //bestHit.t = t;
    //bestHit.position = ray.start + ray.dir * t;
    //bestHit.mat = 0;

    if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
    return bestHit;
}

bool shadowIntersect(Light light, Ray ray, Hit hit) {
    runRay(ray, hits);

    float lengthToLight = length(light.position - hit.position);
    for(int i = 0; i < parts; i++) if(lengthToLight > hits[i].t && hits[i].t > 0.0) return true;

    return false;
}

const float epsilon = 0.01;

vec3 directLight(Light light, Ray ray, Hit hit){
    vec3 outRadiance = vec3(0, 0, 0);
    Ray shadowRay;
    shadowRay.start = hit.position + hit.normal * epsilon;
    shadowRay.dir = normalize(light.position - hit.position);

    float cosTheta = dot(hit.normal, light.position);
    float lengthToLight = length(light.position - hit.position);

    if (cosTheta > 0.0 && !shadowIntersect(light, shadowRay, hit)) {
        outRadiance += light.Le * materials[hit.mat].kd * cosTheta;
        vec3 halfway = normalize(light.position - ray.dir);
        float cosDelta = dot(hit.normal, halfway);
        if (cosDelta > 0.0) outRadiance += light.Le * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);
        outRadiance /= (lengthToLight * lengthToLight);
    }
    outRadiance += materials[hit.mat].ka * light.La; // ambient

    return outRadiance;
}

vec3 trace(Ray ray) {
    vec3 outRadiance = vec3(0, 0, 0);
    Light mainLight = Light(vec3(0,4,10), vec3(30,30,30), vec3(0.06,0.06,0.06));

    Hit hit = firstIntersect(ray);
    if (hit.t < 0) return mainLight.La;

    outRadiance += materials[hit.mat].ka * mainLight.La;
    outRadiance += directLight(mainLight, ray, hit);
    outRadiance += directLight(lightBulb, ray, hit);

    return outRadiance;
}

void main() {
    Ray ray;
    ray.start = wEye + vec3(0,9,0);
    ray.dir = normalize(p - wEye);
    fragmentColor = vec4(trace(ray), 1);
})";

class FullScreenTexturedQuad {
    unsigned int vao;	// vertex array object id and texture id
public:
    void create(){
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
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
    }
};

FullScreenTexturedQuad fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();

    fullScreenTexturedQuad.create();

    shader.create(vertexSource, fragmentSource, "fragmentColor");
    shader.Use();
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(1.0f, 0.5f, 0.8f, 1.0f);							// background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
    frame++;
    static long start = glutGet(GLUT_ELAPSED_TIME);
    long end = glutGet(GLUT_ELAPSED_TIME);
    printf("%d msec\r", (end - start) / frame);

    //shader.setUniform(frame, "frame");
    //int loc =
    scene.setUniform(shader);
    fullScreenTexturedQuad.Draw();

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

