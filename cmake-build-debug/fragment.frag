#version 330 core

const float PI = 3.141592654;
const int parts = 8;

in vec2 texCoord;
uniform float frame;
uniform mat4 viewMat;

out vec4 fragColor;

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

float intersectSphere(vec3 origin, vec3 rayDir, vec3 center, float radius, out vec3 normal) {
    vec3 oc = origin - center;
    float a = dot(rayDir, rayDir);
    float b = 2.0 * dot(oc, rayDir);
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - 4 * a * c;
    if (disc < 0.0) {
        return -1.0;
    }
    float t = (-b - sqrt(disc)) / (2 * a);
    vec3 hitPos = origin + rayDir * t;
    
    normal = normalize(hitPos - center);
    return t;
}

float intersectCylinder(vec3 origin, vec3 rayDir, vec3 center, float radius, float height, out vec3 normal) {
    vec2 oc = origin.xz - center.xz;
    float a = dot(rayDir.xz, rayDir.xz);
    float b = 2.0 * dot(oc, rayDir.xz);
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - 4 * a * c;
    if (disc < 0.0) {
        return -1.0;
    }
    float t1 = (-b - sqrt(disc)) / (2 * a);
    float t2 = (-b + sqrt(disc)) / (2 * a);
    vec3 hitPos1 = origin + rayDir * t1;
    vec3 hitPos2 = origin + rayDir * t2;
    if (hitPos1.y < center.y || hitPos1.y > height + center.y)
        t1 = -1.0;
    if (hitPos2.y < center.y || hitPos2.y > height + center.y)
        t2 = -1.0;
    
    vec3 hitPos;
    float t = combine(t1, t2, hitPos1, hitPos2, hitPos);
    
    normal = hitPos - center;
    normal.y = 0.0;
    normal = normalize(normal);
        
    return t;
}

float intersectPlane(vec3 origin, vec3 rayDir, vec3 point, vec3 normal) {
    return dot(point - origin, normal) / dot(rayDir, normal);
}

float intersectParabole(vec3 origin, vec3 rayDir, vec3 center, float height, out vec3 normal){
    vec3 oc = origin - center;
    float a = dot(rayDir.xz, rayDir.xz);
    float b = 2 * dot(oc.xz, rayDir.xz) - rayDir.y;
    float c = dot(oc.xz, oc.xz) - oc.y;
    float disc = b * b - 4 * a * c;
    if (disc < 0.0) return -1.0;
    
    float t1 = (-b - sqrt(disc)) / (2 * a);
    float t2 = (-b + sqrt(disc)) / (2 * a);

    vec3 hitPos1 = origin + rayDir * t1;
    vec3 hitPos2 = origin + rayDir * t2;

    //if (length(hitPos1 - center) > height) t1 = -1.0;
    //if (length(hitPos2 - center) > height) t2 = -1.0;
    
    vec3 hitPos;
    float t = combine(t1, t2, hitPos1, hitPos2, hitPos);

    if (length(hitPos - center) > height) t = -1.0;
    hitPos = hitPos - center;

    vec3 dx = vec3(1,2 * hitPos.x,0);
    vec3 dz = vec3(0,2 * hitPos.z,1);
    
    normal = cross(dx, dz);
    normal = normalize(normal);
        
    return t;
}

struct pair{
    float shape;
    vec3 normal;
};


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
    pairs[c++] = pair(tTemp, nTemp); // parab
    
    float t = constructShape(pairs, c, normal);

    return combine(t, tPlane, normal, nPlane, normal);
}

void main() {
    float time = frame / 60.0;
    vec3 lightPos = vec3(0, 5, 5);
    
    float fov = PI / 2;
    
    vec3 origin = vec3(0, 2, 10);
    vec3 rayDir = normalize(vec3(texCoord * 2 - 1, -tan(fov / 2.0)));

    vec3 normal = vec3(0,1,0);
    float t = intersectWorld(origin, rayDir, normal);

    if(dot(normal, rayDir) > 0.0) normal *= -1;

    vec3 hitPos = origin + rayDir * t;
    vec3 toLight = lightPos - hitPos;
    float lengthToLight = length(toLight);
    toLight /= lengthToLight;
    if(t > 0.0){
        float cosTheta = max(dot(toLight, normal),0.0);
        vec3 _;
        float lightT = intersectWorld(hitPos + normal * 0.0001, toLight, _);
        float intensity = 40.0;
        if(lightT > 0.0) intensity = 0.0;

        fragColor = vec4((vec3(0.6)) * cosTheta / pow(lengthToLight, 2.0) * intensity,1);
    } else {
        fragColor = vec4(0,0,0,1);
    }
}