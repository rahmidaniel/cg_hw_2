#version 330 core

const float PI = 3.141592654;
const int parts = 9;

struct Material {
    vec3 ka, kd, ks;
    vec3 F0;
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

Hit intersect(const Cylinder obj, const Ray ray) {
    Hit hit;
    hit.mat = 0;
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
    if (hitPos1.y < obj.center.y || hitPos1.y > obj.height + obj.center.y) t1 = -1.0;
    if (hitPos2.y < obj.center.y || hitPos2.y > obj.height + obj.center.y) t2 = -1.0;

    hit.t = combine(t1, t2, hitPos1, hitPos2, hit.position);

    hit.normal = hit.position - obj.center;
    hit.normal.y = 0.0;
    hit.normal = normalize(hit.normal);
    return hit;
}

Hit intersect(const Plane obj, const Ray ray) {
    Hit hit;
    hit.t = -1.f;
    float t = dot(obj.center - ray.start, obj.normal) / dot(ray.dir, obj.normal);

    hit.t = t;
    hit.position = ray.start + ray.dir * hit.t;
    hit.mat = 0;
    if(obj.radius > 0 && hit.position.x * hit.position.x + hit.position.z * hit.position.z > obj.radius * obj.radius) hit.t= -1.0;
    else if(obj.radius == 0.f) hit.mat = 1;

    hit.normal = obj.normal;

    return hit;
}

Hit intersect(const Paraboloid obj, const Ray ray){
    Hit hit;
    hit.mat = 0;
    hit.t = -1;

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
    //if (length(hitPos1 - center) > height) t1 = -1.0;
    //if (length(hitPos2 - center) > height) t2 = -1.0;

    vec3 hitPos;
    float t = combine(t1, t2, hitPos1, hitPos2, hitPos);

    //if (length(hitPos- obj.center) > obj.radius) t = -1.0;
    hit.position = hitPos - obj.center;
    hit.t = t;

    vec3 dx = vec3(1,2 * hit.position.x,0);
    vec3 dz = vec3(0,2 * hit.position.z,1);

    hit.normal = normalize(cross(dx, dz));
    return hit;
}

void checkHit(Hit hit, out Hit bestHit){
    if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) bestHit = hit;
}

Plane plane;
Sphere joints[3];
Cylinder base; Plane baseTop;
Cylinder arms[2];
Paraboloid lamp;

Ray rot1;

void runRay(Ray ray, out Hit hits[parts]){

    // float deskY = -0.4;
    // float jointR = 0.02;
    // float baseR = 0.15, baseH = 0.03;
    // float neckR = 0.015, neckH = 0.25;
    float deskY = -4;
    float jointR = 0.2;
    float baseR = 1.5, baseH = 0.3;
    float neckR = 0.15, neckH = 2.5;
    vec3 planeLevel = vec3(0,deskY,0);

    ////
    vec4 q1 = quat(normalize(vec3(1,4,3)), frame/60.0);
    rot1.start = quatRot(q1, ray.start);
    rot1.dir = quatRot(q1, ray.dir);

    vec4 q2 = quat(normalize(vec3(1,2,3)), frame/60.f);
    Ray rot2;
    rot2.start = quatRot(q2, ray.start);
    rot2.dir = quatRot(q2, ray.dir);
    ////

    int c = 0;
    ////
    plane = Plane(planeLevel, vec3(0,1,0), 0);
    hits[c++] = intersect(plane, ray);

    base = Cylinder(planeLevel, baseR, baseH);
    hits[c++] = intersect(base, ray);

    baseTop = Plane(planeLevel + vec3(0,baseH,0), vec3(0,1,0), baseR);
    hits[c++] = intersect(baseTop, ray);


    joints[0] = Sphere(baseTop.center, jointR);
    joints[0].center = quatRot(q1, joints[0].center);
    Hit inter = intersect(joints[0], rot1);
    inter.normal = quatRot(quatInv(q1), inter.normal); // inverse
    hits[c++] = inter;


    arms[0] = Cylinder(joints[0].center, neckR, neckH);
    inter = intersect(arms[0], rot1);
    inter.normal = quatRot(quatInv(q1), inter.normal); // inverse
    hits[c++] = inter;
    

    joints[1] = Sphere(arms[0].center + vec3(0, neckH, 0), jointR);
    inter = intersect(joints[1], rot1);
    inter.normal = quatRot(quatInv(q1), inter.normal); // inverse
    hits[c++] = inter;

    arms[1] = Cylinder(joints[1].center, neckR, neckH);
    //arm2.center = quatRot(q1, arm2.center);
    inter = intersect(arms[1], rot1);
    inter.normal = quatRot(quatInv(q1), inter.normal); // inverse
    hits[c++] = inter;

    joints[2] = Sphere(arms[1].center + vec3(0, neckH, 0), jointR);
    inter = intersect(joints[2], rot1);
    inter.normal = quatRot(quatInv(q1), inter.normal); // inverse
    hits[c++] = inter;

    lamp = Paraboloid(joints[2].center + vec3(0,joints[2].radius,0), 3);
    inter = intersect(lamp, rot1);
    inter.normal = quatRot(quatInv(q1), inter.normal); // inverse
    hits[c++] = inter;
}

////
Hit firstIntersect(Ray ray) {
    Hit bestHit;
    bestHit.t = -1.f;

    Hit hits[parts];
    runRay(ray, hits);
    for(int i = 0; i < parts; i++) checkHit(hits[i], bestHit);

    if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
    return bestHit;
}
////
bool shadowIntersect(Ray ray) {	// for directional lights
    Hit hits[parts];
    runRay(ray, hits);
    for(int i = 0; i < parts; i++) if(hits[i].t > 0.0) return true;

    return false;
}

const float epsilon = 0.00001;

vec3 directLight(Light light, Ray ray, Hit hit){
    vec3 outRadiance = vec3(0, 0, 0);
    Ray shadowRay;
    shadowRay.start = hit.position + hit.normal * epsilon;
    shadowRay.dir = light.direction;

    Ray lightRay;
    lightRay.start = ray.start + ray.dir * hit.t;
    lightRay.dir = light.direction - hit.position;

    float lenghtToLight = length(lightRay.dir);

    float cosTheta = max(dot(hit.normal, lightRay.dir), 0.0);
    if (cosTheta > 0.0) {
        outRadiance += light.Le * materials[hit.mat].kd * cosTheta;
        
        float intensity = materials[hit.mat].shininess;

        if(shadowIntersect(shadowRay)) intensity = 0;

        //vec3 halfway = normalize(light.direction - ray.dir);
        //float cosDelta = dot(hit.normal, halfway);
        //if (cosDelta > 0.0) outRadiance += weight * light.Le * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);
        outRadiance += cosTheta * light.Le * materials[hit.mat].ks / pow(lenghtToLight, 2.0) * intensity;
    }
    return outRadiance;
}

vec3 trace(Ray ray) {
    vec3 outRadiance = vec3(0, 0, 0);
    Light mainLight = Light(wEye + vec3(0,3,3), vec3(0.03,0.03,0.03), vec3(0.001,0.001,0.001));

    Hit hit = firstIntersect(ray);
    if (hit.t < 0) return mainLight.La;

    outRadiance += materials[hit.mat].ka * mainLight.La;
    outRadiance += directLight(mainLight, ray, hit);

    //vec4 q1 = quat(normalize(vec3(1,4,3)), frame/60.0);

    Light lampLight = Light(lamp.center, vec3(0.02,0.02,0.02), vec3(0,0,0));
    //lampLight.direction = quatRot(q1, lampLight.direction);

    outRadiance += directLight(lampLight, ray, hit);
    
    return outRadiance;
}

// float constructShape(pair pairs[parts], int c, out vec3 normal){
//     float t = combine(pairs[1].shape, pairs[0].shape, pairs[1].normal, pairs[0].normal, normal);
//     for(int i=0; i < c; i++) {
//         t = combine(t, pairs[i].shape, normal, pairs[i].normal, normal);
//     }
//     return t;
// }

void main() {
    Ray ray;
    ray.start = wEye + vec3(0,3,25);
    ray.dir = normalize(p - wEye);
    fragmentColor = vec4(trace(ray), 1);
}