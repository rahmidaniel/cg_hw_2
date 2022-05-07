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
public:
    vec3* center;
    float radius;

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
    Sphere(vec3* _center, float _radius) {
        center = _center;
        radius = _radius;
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
        return hit;
    }
};

struct Paraboloid : public Intersectable {
    Paraboloid(vec3* _center, float _radius) {
        center = _center;
        radius = _radius;
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
        return hit;
    }
};

struct Cylinder : public Intersectable {
    float height;

    Cylinder(vec3 *_center, float _radius, float _height) {
        center = _center;
        radius = _radius;
        height = _height;
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
        return hit;
    }
};

struct Plane : public Intersectable{
    vec3 normal;
    Plane(vec3 *_center, float _radius, const vec3 &_normal) {
        center = _center;
        radius = _radius;
        normal = _normal;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;
        float t = dot(*center - ray.start, normal) / dot(ray.dir, normal);
        if(t < 0.f) return hit;

        hit.position = ray.start + ray.dir * t;
        if(radius != 0 && hit.position.x * hit.position.x + hit.position.z * hit.position.z > radius * radius) t = -1.f;

        hit.t = t;
        hit.normal = normal;
        return hit;
    }
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
        vec3 d = eye - lookat;
        //Quaternion q(normalize(vec3(1,6,3)), frame/60.f);
        //eye = quatRot(q, lookat);
        eye = vec3(d.x * cos(dt) + d.z * sin(dt), d.y, -d.x * sin(dt) + d.z * cos(dt)) + lookat;
        set(eye, lookat, up, fov);
    }
};

struct Light {
    vec3 direction;
    vec3 Le, La;
    Light(vec3 _direction, vec3 _Le, vec3 _La) {
        direction = normalize(_direction);
        Le = _Le;
        La = _La;
    }
};

struct Shape{
    float radius;
    vec3 center;
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

    void setUniformLight(Light* light) {
        setUniform(light->La, "light.La");
        setUniform(light->Le, "light.Le");
        setUniform(light->direction, "light.direction");
    }

    void setUniformCamera(const Camera& camera) {
        setUniform(camera.eye, "wEye");
        setUniform(camera.lookat, "wLookAt");
        setUniform(camera.right, "wRight");
        setUniform(camera.up, "wUp");
    }

    void setUniformObjects(const std::vector<Intersectable*>& shapes) {
        char name[256];
        for (unsigned int o = 0; o < shapes.size()-1; o++) {
            sprintf(name, "shapes[%d].center", o);  setUniform(*shapes[o]->center, name);
            sprintf(name, "shapes[%d].radius", o);  setUniform(shapes[o]->radius, name);
        }
//        sprintf(name, "parabola.center");  setUniform(*shapes[8]->center, name);
//        sprintf(name, "parabola.radius");  setUniform(shapes[8]->radius, name);
    }
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

class Scene {
    std::vector<Intersectable *> objects;
    std::vector<Light *> lights;
    std::vector<Material *> materials;
    Camera camera;
    //vec3 La;

    std::vector<vec3*> nodes;
public:
    void build() {
        vec3 eye = vec3(0, 0, 2), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
        float fov = 45 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        vec3 kd(0.1f, 0.2f, 0.9f), ks(0.6, 0.6, 0.6);
        vec3 kd2(0.4f, 0.2f, 0.1f), ks2(1, 1, 1);
        materials.push_back(new Material(kd, ks, 60));
        materials.push_back(new Material(kd2, ks2, 30));

        float deskY = -0.4;
        float jointR = 0.02;
        float baseR = 0.15, baseH = 0.03;
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
        objects.push_back(new Plane(nodes[0], 0.f, vec3(0, 1, 0)));
        //1,2
        objects.push_back(new Cylinder(nodes[0], baseR, baseH));
        objects.push_back(new Plane(nodes[1], baseR, vec3(0, 1, 0)));
        //3,4
        objects.push_back(new Sphere(nodes[2], jointR));
        objects.push_back(new Cylinder(nodes[2], neckR, neckH));
        //5,6
        objects.push_back(new Sphere(nodes[3], jointR));
        objects.push_back(new Cylinder(nodes[3], neckR, neckH));
        //7,8
        objects.push_back(new Sphere(nodes[4], jointR));
        objects.push_back(new Paraboloid(nodes[4], jointR));

        // light
        //lights.push_back(new Light(vec3(0,deskY + baseH + neckH * 2,0.2), Le * 0.5f));
//        vec3 Le(0.5, 0.5, 0.5), La(0.01f, 0.01f, 0.01f);
//        lights.push_back(new Light(eye + vec3(0,0,10), Le, La));
        //lights.push_back(new Light(lightDirection, Le, La));
    }
    void setUniform(Shader& shader) {
        shader.setUniformObjects(objects);
        shader.setUniformMaterials(materials);
        //shader.setUniformLight(lights[0]);
        shader.setUniformCamera(camera);
    }

    void animate(float dt){
        //camera.animate(dt);
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
const char *fragmentSource = R"(
#version 330 core

const float PI = 3.141592654;
const int parts = 9;

struct Material {
    vec3 ka, kd, ks;
    float  shininess;
};

struct Light {
    vec3 direction;
    vec3 Le, La;
};

struct Sphere {
    vec3 center;
    float radius;
};
struct Paraboloid {
    vec3 center;
    float radius;
};
struct Cylinder {
    vec3 center;
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

const int nMaxObjects = 500;

uniform vec3 wEye;
uniform Light light;
uniform Material materials[2];  // diffuse, specular, ambient ref
uniform Sphere joints[3];
uniform Paraboloid lamp;
uniform Cylinder arms[2];
uniform Cylinder base;
uniform Plane planes[2];

uniform Shape shapes[1 + 2 + 3 + 2 + 1];

in  vec3 p;					// center on camera window corresponding to the pixel
out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

vec4 quat(vec3 axis, float angle) {
    return vec4(axis * sin(angle / 2), cos(angle / 2));
}

vec4 quatInv(vec4 q1) {
    return vec4(-q1.xyz, q1.w);
}

vec4 quatMul(vec4 q1, vec4 q2) {
    return vec4(
    q1.w * q2.xyz + q2.w * q1.xyz + cross(q1.xyz, q2.xyz),
    q1.w * q2.w - dot(q1.xyz, q2.xyz)
    );
}

vec3 quatRot(vec4 q1, vec3 p) {
    vec4 qInv = quatInv(q1);
    return quatMul(quatMul(q1, vec4(p, 0)), qInv).xyz;
}

vec4 Rotate(vec3 u, vec3 v){
    vec3 h = u + v / length(u+v);
    return vec4(dot(u,h), cross(u,h));
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
    hit.t = -1.f;
    vec3 oc = ray.start - obj.center;
    float a = dot(ray.dir, ray.dir);
    float b = 2.0 * dot(oc, ray.dir);
    float c = dot(oc, oc) - obj.radius * obj.radius;
    float disc = b * b - 4 * a * c;
    if (disc < 0.0) return hit;

    hit.t = (-b - sqrt(disc)) / (2 * a);
    hit.position = ray.start + ray.dir * t;
    hit.normal = normalize(hit.position - obj.center);

    return hit;
}

Hit intersect(const Cylinder obj, const Ray ray) {
    Hit hit;
    hit.t = -1.f;
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
    if (hitPos1.y < center.y || hitPos1.y > obj.height + obj.center.y) t1 = -1.0;
    if (hitPos2.y < center.y || hitPos2.y > obj.height + obj.center.y) t2 = -1.0;

    hit.t = combine(t1, t2, hitPos1, hitPos2, hit.position);

    hit.normal = normalize(hit.position - obj.center);
    hit.normal.y = 0.0;
    return hit;
}

Hit intersect(const Plane obj, const Ray ray) {
    Hit hit;
    hit.t = -1.f;
    float t = dot(obj.center - origin, obj.normal) / dot(ray.dir, obj.normal);

    if(obj.radius == 0.f) t = -1.f;
    hit.t = t;
    hit.position = ray.start + ray.dir * t1;
    hit.normal = obj.normal;

    return hit;
}

Hit intersect(const Paraboloid obj, const Ray ray){
    Hit hit;
    hit.t = -1.f;

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

    //if (length(hitPos1 - center) > height) t1 = -1.0;
    //if (length(hitPos2 - center) > height) t2 = -1.0;

    vec3 hitPos;
    float t = combine(t1, t2, hitPos1, hitPos2, hitPos);

    if (length(hitPos- obj.center) > obj.radius) t = -1.0;
    hit.position = hitPos - obj.center;
    hit.t = t;

    vec3 dx = vec3(1,2 * hit.position.x,0);
    vec3 dz = vec3(0,2 * hit.position.z,1);

    hit.normal = normalize(cross(dx, dz));
    return hit;
}

struct pair{
    float shape;
    vec3 normal;
};
void checkHit(Hit hit, out Hit bestHit){
    if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)){
        bestHit = hit;
        bestHit.mat = 0;
    }
}

////
Hit firstIntersect(Ray ray) {
    Hit bestHit;
    bestHit.t = -1;
    bestHit.mat = 1;
    ////
    Plane p = Plane(shapes[0].center, vec3(0,1,0), shapes[0].radius);
    checkHit(intersect(p), bestHit);

    Cylinder base = Cylinder(shapes[1].center, shapes[1].radius, 0.03);
    checkHit(intersect(base), bestHit);

    Plane baseTop = Plane(shapes[2].center, vec3(0,1,0), shapes[2].radius);
    checkHit(intersect(baseTop), bestHit);

    Sphere joint1 = Sphere(shapes[3].center, shapes[3].radius);
    checkHit(intersect(joint1), bestHit);

    Cylinder arm1 = Cylinder(shapes[3].center, shapes[3].radius, 0.25);
    checkHit(intersect(arm1), bestHit);

    Sphere joint2 = Sphere(shapes[4].center, shapes[4].radius);
    checkHit(intersect(joint2), bestHit);

    Cylinder arm2 = Cylinder(shapes[5].center, shapes[5].radius, 0.25);
    checkHit(intersect(arm2), bestHit);

    Sphere joint3 = Sphere(shapes[6].center, shapes[6].radius);
    checkHit(intersect(joint3), bestHit);

    Paraboloid lamp = Paraboloid(shapes[7].center, shapes[7].radius);
    checkHit(intersect(lamp), bestHit);

    ///
    if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
    return bestHit;
}
////
bool shadowIntersect(Ray ray) {	// for directional lights
    for (int o = 0; o < parts; o++) if (intersect(pairs[o], ray).t > 0) return true; //  hit.t < 0 if no intersection
    return false;
}

vec3 Fresnel(vec3 F0, float cosTheta) {
    return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
}
const float epsilon = 0.0001f;

vec3 trace(Ray ray) {
    vec3 weight = vec3(1, 1, 1);
    vec3 outRadiance = vec3(0, 0, 0);

    Hit hit = firstIntersect(ray);
    if (hit.t < 0) return weight * light.La;

    outRadiance += weight * materials[hit.mat].ka * light.La;
    Ray shadowRay;
    shadowRay.start = hit.position + hit.normal * epsilon;
    shadowRay.dir = light.direction;
    float cosTheta = dot(hit.normal, light.direction);
    if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
        outRadiance += weight * light.Le * materials[hit.mat].kd * cosTheta;
        vec3 halfway = normalize(-ray.dir + light.direction);
        float cosDelta = dot(hit.normal, halfway);
        if (cosDelta > 0) outRadiance += weight * light.Le * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);
    }
    return outRadiance;
}

float constructShape(pair pairs[parts], int c, out vec3 normal){
    float t = combine(pairs[1].shape, pairs[0].shape, pairs[1].normal, pairs[0].normal, normal);
    for(int i=0; i < c; i++) {
        t = combine(t, pairs[i].shape, normal, pairs[i].normal, normal);
    }
    return t;
}

float intersectWorld(vec3 origin, vec3 rayDir, out vec3 normal) {
    float time = frame / 60.0;

    float floorPos = -0.2;
    float baseR = 2;
    float baseH = 0.2;
    float basePos = 0;
    float jointR = 0.3;
    float neckR = 0.15;
    float neckH = 2.5;
    float parab = 1;

    pair pairs[parts];
    int c = 0;

    // Rotate with this
    vec4 q1 = quat(normalize(vec3(1,6,3)), time);
    vec3 rotOrigin1 = quatRot(q1, origin);
    vec3 rotRayDir1 = quatRot(q1, rayDir);

    vec4 q2 = quat(normalize(vec3(0,4,3)), time);
    vec3 rotOrigin2 = quatRot(q2, origin);
    vec3 rotRayDir2 = quatRot(q2, rayDir);

    vec4 q3 = quatMul(q1,q2);
    vec3 rotOrigin3 = quatRot(q3, origin);
    vec3 rotRayDir3 = quatRot(q3, rayDir);

    vec3 nPlane = vec3(0,1,0);
    float tPlane = intersectPlane(origin, rayDir, vec3(0,floorPos,0), nPlane);

    vec3 nTemp;
    float tTemp;

    tTemp = intersectCylinder(origin, rayDir, vec3(0, basePos-0.1, 0), baseR, baseH, nTemp);
    pairs[c++] = pair(tTemp, nTemp); // base

    nTemp = vec3(0,1,0);
    tTemp = intersectPlane(origin, rayDir, vec3(0, basePos-0.1 + baseH, 0), nTemp);

    vec3 hitPosPlane = origin + rayDir * tTemp;
    if(hitPosPlane.x * hitPosPlane.x + hitPosPlane.z * hitPosPlane.z > baseR * baseR) tTemp = -1.0;
    pairs[c++] = pair(tTemp, nTemp); // cyl top

    nTemp = vec3(0);
    tTemp = intersectSphere(origin, rayDir, vec3(0,basePos,0), jointR, nTemp);
    pairs[c++] = pair(tTemp, nTemp); // shp1

    nTemp = vec3(0);
    tTemp = intersectCylinder(rotOrigin1, rotRayDir1, vec3(0,basePos,0), neckR, neckH, nTemp);
    nTemp = quatRot(quatInv(q1), nTemp); // inverse
    pairs[c++] = pair(tTemp, nTemp); // neck1

    nTemp = vec3(0);
    tTemp = intersectSphere(rotOrigin1, rotRayDir1, vec3(0, neckH, 0), jointR, nTemp);
    nTemp = quatRot(quatInv(q1), nTemp); // inverse
    pairs[c++] = pair(tTemp, nTemp); // shp2

    //nTemp = vec3(0);
    //vec4 q = Rotate(vec3(0, neckH, 0), )
    tTemp = intersectCylinder(rotOrigin2, rotRayDir2, vec3(0, neckH, 0), neckR, neckH, nTemp);
    nTemp = quatRot(quatInv(q2), nTemp); // inverse
    pairs[c++] = pair(tTemp, nTemp); // heck2

    nTemp = vec3(0);
    tTemp = intersectSphere(rotOrigin1, rotRayDir1, vec3(0, neckH * 2, 0), jointR, nTemp);
    nTemp = quatRot(quatInv(q1), nTemp); // inverse
    pairs[c++] = pair(tTemp, nTemp); // shp1

    tTemp = intersectParabole(rotOrigin1, rotRayDir1, vec3(0,2,4), parab, nTemp);
    nTemp = quatRot(quatInv(q1), nTemp); // inverse
    //pairs[c++] = pair(tTemp, nTemp); // parab

    float t = constructShape(pairs, c, normal);

    return combine(t, tPlane, normal, nPlane, normal);
}

void main() {
    //    float time = frame / 60.0;
    //    vec3 lightPos = vec3(0, 5, 5);
    //
    //    float fov = PI / 2;
    //
    //    vec3 origin = vec3(0, 2, 10);
    //    vec3 rayDir = normalize(vec3(texCoord * 2 - 1, -tan(fov / 2.0)));
    //
    //    vec3 normal = vec3(0,1,0);
    //    float t = intersectWorld(origin, rayDir, normal);
    //
    //    if(dot(normal, rayDir) > 0.0) normal *= -1;
    //
    //    vec3 hitPos = origin + rayDir * t;
    //    vec3 toLight = lightPos - hitPos;
    //    float lengthToLight = length(toLight);
    //    toLight /= lengthToLight;
    //    if(t > 0.0){
    //        float cosTheta = max(dot(toLight, normal),0.0);
    //        vec3 _;
    //        float lightT = intersectWorld(hitPos + normal * 0.0001, toLight, _);
    //        float intensity = 40.0;
    //        if(lightT > 0.0) intensity = 0.0;
    //
    //        fragColor = vec4((vec3(0.6)) * cosTheta / pow(lengthToLight, 2.0) * intensity,1);
    //    } else {
    //        fragColor = vec4(0,0,0,1);
    //    }
    Ray ray;
    ray.start = wEye;
    ray.dir = normalize(p - wEye);
    fragmentColor = vec4(trace(ray), 1);
}
)";

// LEADAS ELOTT VEDD KI //TODO
#include <fstream>
#include <sstream>
// VEGE

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

    // LEADAS ELOTT VEDD KI TODO

    std::string newVertexSrc;
    std::string newFragmentSrc;
    std::string line;
    std::ifstream vfile("vertex.vert");
    while (std::getline(vfile, line)) {
        newVertexSrc += line + "\n";
    }
    vfile.close();
    std::ifstream ffile("fragment.frag");
    while (std::getline(ffile, line)) {
        newFragmentSrc += line + "\n";
    }
    ffile.close();

    shader.create(newVertexSrc.c_str(), newFragmentSrc.c_str(), "fragmentColor");
    shader.Use();

    // create program for the GPU
    //shader.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(1.0f, 0.5f, 0.8f, 1.0f);							// background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
    //static int frame = 0;
    frame++;
    static long start = glutGet(GLUT_ELAPSED_TIME);
    long end = glutGet(GLUT_ELAPSED_TIME);
    printf("%d msec\r", (end - start) / frame);

    // LEADAS ELOTT VEDD KI TODO

    std::string newVertexSrc;
    std::string newFragmentSrc;
    std::string line;
    std::ifstream vfile("vertex.vert");
    while (std::getline(vfile, line)) {
        newVertexSrc += line + "\n";
    }
    vfile.close();
    std::ifstream ffile("fragment.frag");
    while (std::getline(ffile, line)) {
        newFragmentSrc += line + "\n";
    }
    ffile.close();

    shader.create(newVertexSrc.c_str(), newFragmentSrc.c_str(), "fragmentColor");
    shader.Use();

    shader.setUniform(frame, "frame");
    scene.setUniform(shader);
    fullScreenTexturedQuad.Draw();

    glutSwapBuffers();									// exchange the two buffers
    glutPostRedisplay();
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

