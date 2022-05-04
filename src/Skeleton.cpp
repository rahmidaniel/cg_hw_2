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

const char *const vertexSource = R"()";

const char *const fragmentSource = R"()";

GPUProgram gpuProgram(false);
unsigned int vao;
int frame = 0;

const float PI = 3.141592654;
const int parts = 8;

vec4 quat(vec3 axis, float angle) {
    axis = axis * sin(angle / 2);
    return {axis.x, axis.y, axis.z, cos(angle / 2)};
}

vec4 quatInv(vec4 q1) {
    q1.w *= -1;
    return q1;
}

vec4 quatMul(vec4 q1, vec4 q2) {
    return vec4(
            q1.w * q2.xyz + q2.w * q1.xyz + cross(q1.xyz, q2.xyz),
            q1.w * q2.w - dot(q1.xyz, q2.xyz)
    );
}

vec3 quatRot(vec4 q1, vec3 p) {
    vec4 qInv = quatInv(q1);
    return quatMul(quatMul(q1, vec4(p.x, p.y, p.z, 0)), qInv).xyz;
}

vec4 Rotate(vec3 u, vec3 v){
    vec3 h = u + v / length(u+v);
    vec3 c = cross(u,h);
    return {dot(u,h), c.x, c.y, c.z};
}

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

float intersectSphere(vec3 origin, vec3 rayDir, vec3 center, float radius, vec3& normal) {
    vec3 oc = origin - center;
    float a = dot(rayDir, rayDir);
    float b = 2.f * dot(oc, rayDir);
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

float intersectCylinder(vec3 origin, vec3 rayDir, vec3 center, float radius, float height, vec3& normal) {
    vec2 rayXZ = vec2(rayDir.x, rayDir.z);
    vec2 oc = vec2(origin.x, origin.z) - vec2(center.x, center.z);
    float a = dot(rayXZ, rayXZ);
    float b = 2.f * dot(oc, rayXZ);
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

float intersectParabole(vec3 origin, vec3 rayDir, vec3 center, float height, vec3& normal){
    vec3 oc = origin - center;
    vec2 rayXZ = vec2(rayDir.x, rayDir.z);
    vec2 ocXZ = vec2(oc.x, oc.z);
    float a = dot(rayXZ, rayXZ);
    float b = 2 * dot(ocXZ, rayXZ) - rayDir.y;
    float c = dot(ocXZ, ocXZ) - oc.y;
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

    explicit pair(float t = 0, vec3 n = 0){
        normal = n;
        shape = t;
    }
};


float constructShape(pair pairs[parts], int c, vec3& normal){
    float t = combine(pairs[1].shape, pairs[0].shape, pairs[1].normal, pairs[0].normal, normal);
    for(int i=0; i < c; i++) {
        t = combine(t, pairs[i].shape, normal, pairs[i].normal, normal);
    }
    return t;
}

float intersectWorld(vec3 origin, vec3 rayDir, vec3& normal) {
    float time = frame / 60.f;

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

void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    unsigned int vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    float vertices[] = {-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

    gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// LEADAS ELOTT VEDD KI
#include <fstream>
#include <sstream>
// VEGE

void onDisplay() {
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    // LEADAS ELOTT VEDD KI

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

    GPUProgram gpuProgram(false);
    gpuProgram.create(newVertexSrc.c_str(), newFragmentSrc.c_str(), "outColor");

    // VEGE
    int loc = glGetUniformLocation(gpuProgram.getId(), "frame");
    glUniform1f(loc, (float)frame);

    frame++;

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glutSwapBuffers();
    glutPostRedisplay();
}

void onKeyboard(unsigned char key, int pX, int pY) {
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onMouse(int button, int state, int pX, int pY) {
}

void onIdle() {
}

