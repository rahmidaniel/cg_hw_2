#version 330 core

const float PI = 3.141592654;
const int parts = 8;

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
    
    normal = normalize(cross(dx, dz));

    return hit;
}

struct pair{
    float shape;
    vec3 normal;
};
////
Hit firstIntersect(Ray ray) {
    Hit bestHit;
    bestHit.t = -1;
    for (int o = 0; o < parts; o++) {
        Hit hit = intersect(pairs[o], ray);
        if (o == 0) hit.mat = 0;
        else hit.mat = 1;
        if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
    }
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
    if (materials[hit.mat] == 1) {
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
    };

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