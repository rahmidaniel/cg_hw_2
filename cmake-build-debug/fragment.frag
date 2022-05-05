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
struct Shape{
    vec3 center;
    float radius;
};

uniform vec3 wEye;
uniform Light light;
uniform Material materials[2];  // diffuse, specular, ambient ref
uniform int frame;

uniform Shape shapes[parts];
uniform Paraboloid parabola;

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
    if(obj.radius != 0.f && hit.position.x * hit.position.x + hit.position.z * hit.position.z > obj.radius * obj.radius) hit.t= -1.0;
    else if(obj.radius == 0.f) hit.mat = 1;

    hit.normal = obj.normal;

    return hit;
}

Hit intersect(const Paraboloid obj, const Ray ray){
    Hit hit;
    hit.mat = 0;
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

////
Hit firstIntersect(Ray ray) {
    Hit bestHit;
    bestHit.t = -1.f;
    ////
    vec4 q1 = quat(normalize(vec3(1,4,3)), frame/60.0);
    Ray rot1;
    rot1.start = quatRot(q1, ray.start);
    rot1.dir = quatRot(q1, ray.dir);

    vec4 q2 = quat(normalize(vec3(1,2,3)), frame/60.f);
    Ray rot2;
    rot2.start = quatRot(q2, ray.start);
    rot2.dir = quatRot(q2, ray.dir);


    ////
    Plane p = Plane(shapes[0].center, vec3(0,1,0), shapes[0].radius);
    checkHit(intersect(p, ray), bestHit);

    Cylinder base = Cylinder(shapes[1].center, shapes[1].radius, 0.03);
    checkHit(intersect(base, ray), bestHit);

    Plane baseTop = Plane(shapes[2].center, vec3(0,1,0), shapes[1].radius);
    checkHit(intersect(baseTop, ray), bestHit);


    Sphere joint1 = Sphere(shapes[3].center, shapes[3].radius);
    joint1.center = quatRot(q1, shapes[3].center);
    Hit inter = intersect(joint1, rot1);
    inter.normal = quatRot(quatInv(q1), inter.normal); // inverse
    checkHit(inter, bestHit);


    Cylinder arm1 = Cylinder(joint1.center, shapes[4].radius, 0.25);
    inter = intersect(arm1, rot1);
    inter.normal = quatRot(quatInv(q1), inter.normal); // inverse
    checkHit(intersect(arm1, rot1), bestHit);
    

    Sphere joint2 = Sphere(arm1.center - shapes[5].center * 2, shapes[5].radius);
    inter = intersect(joint2, rot1);
    inter.normal = quatRot(quatInv(q1), inter.normal); // inverse
    checkHit(inter, bestHit);

    Cylinder arm2 = Cylinder(joint2.center, shapes[6].radius, 0.25);
    arm2.center = quatRot(q2, arm2.center);
    inter = intersect(arm2, rot1);
    inter.normal = quatRot(quatInv(q1), inter.normal); // inverse
    checkHit(inter, bestHit);

    Sphere joint3 = Sphere(arm2.center - shapes[6].center * 2, shapes[7].radius);
    inter = intersect(joint3, rot1);
    inter.normal = quatRot(quatInv(q1), inter.normal); // inverse
    checkHit(inter, bestHit);

    Paraboloid lamp = Paraboloid(joint3.center, 0.2);
    inter = intersect(lamp, rot1);
    inter.normal = quatRot(quatInv(q1), inter.normal); // inverse
    checkHit(inter, bestHit);
    //checkHit(intersect(lamp, rot1), bestHit);

    ///
    if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
    return bestHit;
}
////
bool shadowIntersect(Ray ray) {	// for directional lights
    Plane p = Plane(shapes[0].center, vec3(0,1,0), shapes[0].radius);
    if(intersect(p, ray).t > 0.0) return true;

    Cylinder base = Cylinder(shapes[1].center, shapes[1].radius, 0.03);
    if(intersect(base, ray).t > 0.0) return true;

    Plane baseTop = Plane(shapes[2].center, vec3(0,1,0), shapes[2].radius);
    if(intersect(baseTop, ray).t > 0.0) return true;

    Sphere joint1 = Sphere(shapes[3].center, shapes[3].radius);
    if(intersect(joint1, ray).t > 0.0) return true;

    Cylinder arm1 = Cylinder(shapes[4].center, shapes[4].radius, 0.25);
    if(intersect(arm1, ray).t > 0.0) return true;

    Sphere joint2 = Sphere(shapes[5].center, shapes[5].radius);
    if(intersect(joint2, ray).t > 0.0) return true;

    Cylinder arm2 = Cylinder(shapes[6].center, shapes[6].radius, 0.25);
    if(intersect(arm2, ray).t > 0.0) return true;

    Sphere joint3 = Sphere(shapes[7].center, shapes[7].radius);
    if(intersect(joint3, ray).t > 0.0) return true;

    Paraboloid lamp = Paraboloid(shapes[8].center, shapes[8].radius);
    if(intersect(lamp, ray).t > 0.0) return true;

    return false;
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

// float constructShape(pair pairs[parts], int c, out vec3 normal){
//     float t = combine(pairs[1].shape, pairs[0].shape, pairs[1].normal, pairs[0].normal, normal);
//     for(int i=0; i < c; i++) {
//         t = combine(t, pairs[i].shape, normal, pairs[i].normal, normal);
//     }
//     return t;
// }

//float intersectWorld(vec3 origin, vec3 rayDir, out vec3 normal) {
//    float time = frame / 60.0;
//
//    float floorPos = -0.2;
//    float baseR = 2;
//    float baseH = 0.2;
//    float basePos = 0;
//    float jointR = 0.3;
//    float neckR = 0.15;
//    float neckH = 2.5;
//    float parab = 1;
//
//    pair pairs[parts];
//    int c = 0;
//
//    // Rotate with this
//    vec4 q1 = quat(normalize(vec3(1,6,3)), time);
//    vec3 rotOrigin1 = quatRot(q1, origin);
//    vec3 rotRayDir1 = quatRot(q1, rayDir);
//
//    vec4 q2 = quat(normalize(vec3(0,4,3)), time);
//    vec3 rotOrigin2 = quatRot(q2, origin);
//    vec3 rotRayDir2 = quatRot(q2, rayDir);
//
//    vec4 q3 = quatMul(q1,q2);
//    vec3 rotOrigin3 = quatRot(q3, origin);
//    vec3 rotRayDir3 = quatRot(q3, rayDir);
//
//    vec3 nPlane = vec3(0,1,0);
//    float tPlane = intersectPlane(origin, rayDir, vec3(0,floorPos,0), nPlane);
//
//    vec3 nTemp;
//    float tTemp;
//
//    tTemp = intersectCylinder(origin, rayDir, vec3(0, basePos-0.1, 0), baseR, baseH, nTemp);
//    pairs[c++] = pair(tTemp, nTemp); // base
//
//    nTemp = vec3(0,1,0);
//    tTemp = intersectPlane(origin, rayDir, vec3(0, basePos-0.1 + baseH, 0), nTemp);
//
//    vec3 hitPosPlane = origin + rayDir * tTemp;
//    if(hitPosPlane.x * hitPosPlane.x + hitPosPlane.z * hitPosPlane.z > baseR * baseR) tTemp = -1.0;
//    pairs[c++] = pair(tTemp, nTemp); // cyl top
//
//    nTemp = vec3(0);
//    tTemp = intersectSphere(origin, rayDir, vec3(0,basePos,0), jointR, nTemp);
//    pairs[c++] = pair(tTemp, nTemp); // shp1
//
//    nTemp = vec3(0);
//    tTemp = intersectCylinder(rotOrigin1, rotRayDir1, vec3(0,basePos,0), neckR, neckH, nTemp);
//    nTemp = quatRot(quatInv(q1), nTemp); // inverse
//    pairs[c++] = pair(tTemp, nTemp); // neck1
//
//    nTemp = vec3(0);
//    tTemp = intersectSphere(rotOrigin1, rotRayDir1, vec3(0, neckH, 0), jointR, nTemp);
//    nTemp = quatRot(quatInv(q1), nTemp); // inverse
//    pairs[c++] = pair(tTemp, nTemp); // shp2
//
//    //nTemp = vec3(0);
//    //vec4 q = Rotate(vec3(0, neckH, 0), )
//    tTemp = intersectCylinder(rotOrigin2, rotRayDir2, vec3(0, neckH, 0), neckR, neckH, nTemp);
//    nTemp = quatRot(quatInv(q2), nTemp); // inverse
//    pairs[c++] = pair(tTemp, nTemp); // heck2
//
//    nTemp = vec3(0);
//    tTemp = intersectSphere(rotOrigin1, rotRayDir1, vec3(0, neckH * 2, 0), jointR, nTemp);
//    nTemp = quatRot(quatInv(q1), nTemp); // inverse
//    pairs[c++] = pair(tTemp, nTemp); // shp1
//
//    tTemp = intersectParabole(rotOrigin1, rotRayDir1, vec3(0,2,4), parab, nTemp);
//    nTemp = quatRot(quatInv(q1), nTemp); // inverse
//    //pairs[c++] = pair(tTemp, nTemp); // parab
//
//    float t = constructShape(pairs, c, normal);
//
//    return combine(t, tPlane, normal, nPlane, normal);
//}

void main() {
    Ray ray;
    ray.start = wEye;
    ray.dir = normalize(p - wEye);
    fragmentColor = vec4(trace(ray), 1);
}