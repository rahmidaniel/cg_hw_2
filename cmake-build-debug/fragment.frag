#version 330 core

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

uniform Shape shapes[parts];

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

    vec3 hitPos;
    float t = combine(t1, t2, hitPos1, hitPos2, hitPos);

    hit.position = hitPos - obj.center;
    hit.t = t;

    vec3 dx = vec3(1,2 * hit.position.x,0);
    vec3 dz = vec3(0,2 * hit.position.z,1);

    hit.normal = normalize(cross(dx, dz));

    return hit;
}

Plane plane;
Sphere joints[3];
Cylinder base; Plane baseTop;
Cylinder arms[2];
Paraboloid lamp;
Light lightBulb;

void animate(){
    float time = frame/60.0;
    vec4 q1 = quat(normalize(vec3(1,5,3)), time);
    vec4 q2 = quat(normalize(vec3(1,5,3)), time);
    vec4 q3 = quat(normalize(vec3(1,5,3)), time);

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

void build(){
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

    lightBulb = Light(lamp.center + lamp.dir / 2.0, vec3(50,50,50), vec3(0,0,0));
}

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
    float t = combine(hits[1].t, hits[0].t, hits[1].normal, hits[0].normal, bestHit.normal);
    for(int i = 0; i < parts; i++) {
        t = combine(t, hits[i].t, bestHit.normal, hits[i].normal, bestHit.normal);
        //checkHit(hits[i], bestHit);
    }
    bestHit.t = t;
    bestHit.position = ray.start + ray.dir * t;
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

const float epsilon = 0.001;

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
        outRadiance /= lengthToLight * lengthToLight;
    }
    outRadiance += materials[hit.mat].ka * light.La; // ambient

    return outRadiance;
}

vec3 trace(Ray ray) {
    vec3 outRadiance = vec3(0, 0, 0);
    Light mainLight = Light(wEye + vec3(0,9,9), vec3(30,30,30), vec3(0.01,0.01,0.01));

    build();
    animate();
    //runRay(ray, hits);

    Hit hit = firstIntersect(ray);
    if (hit.t < 0) return mainLight.La;

    outRadiance += materials[hit.mat].ka * mainLight.La;
    outRadiance += directLight(mainLight, ray, hit);
    outRadiance += directLight(lightBulb, ray, hit);
    
    return outRadiance;
}

void main() {
    Ray ray;
    ray.start = wEye + vec3(0,3,25);
    ray.dir = normalize(p - wEye);
    fragmentColor = vec4(trace(ray), 1);
}