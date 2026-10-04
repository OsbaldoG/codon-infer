#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("--- Bio Strata Engine V0.1 ---\n");
    printf("Iniciando carga de pesos del modelo ESM-2...\n\n");

    FILE *file = fopen("esm2_pesos.bin", "rb");
    if (file == NULL) {
        printf("Error: No se pudo encontrar el archivo esm2_pesos.bin\n");
        return 1;
    }

    int num_pesos_a_leer = 10;
    float *pesos_iniciales = (float*)malloc(num_pesos_a_leer * sizeof(float));

    if (pesos_iniciales == NULL) {
        printf("Error: No hay memoria suficiente.\n");
        return 1;
    }

    size_t leidos = fread(pesos_iniciales, sizeof(float), num_pesos_a_leer, file);

    if (leidos == num_pesos_a_leer) {
        printf("¡Éxito! Se ha conectado con la matriz de pesos.\n");
        printf("Los primeros %d valores matemáticos del modelo son:\n", num_pesos_a_leer);
        for (int i = 0; i < num_pesos_a_leer; i++) {
            printf("Peso [%d]: %f\n", i, pesos_iniciales[i]);
        }
    } else {
        printf("Error al leer los pesos.\n");
    }

    fclose(file);
    free(pesos_iniciales);
    printf("\nPrueba de lectura superada.\n");
    return 0;
}
