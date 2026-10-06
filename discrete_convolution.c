#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ТЗ не специфицирует коды ошибок кроме случая "Если какое-то из имён файлов не задано".
// Для единообразия возвращаем 10 при любой критической ошибке (файл/память и тп).

typedef struct 
{
    uint32_t H;     // высота матрицы
    uint32_t W;     // ширина матрицы
    uint8_t* start; // указатель на начало памяти
} matrix;

typedef struct
{
    uint16_t DH;    // высота матрицы
    uint16_t DW;    // ширина матрицы
    int8_t* start;  // указатель на начало памяти
} matrix_kernel;

// Чтение данных из файла. Функция сама выделяет память под структуры A, B, C, D.
// При любой ошибке (открытие файла, чтение, выделение памяти, неверный формат)
// гарантирует закрытие файла и освобождение уже выделенных ресурсов.
// Возвращает 0 при успехе, 10 при ошибке.
int read_data(const char* fname, matrix* A, matrix* B, matrix* C, matrix_kernel* D) 
{
    FILE* file = fopen(fname, "rb");

    if (file == NULL) 
    {
        return 10;
    }

    if (fread(&(A->H), sizeof(uint32_t), 1, file) != 1
    ||  fread(&(A->W), sizeof(uint32_t), 1, file) != 1)
    {
        fclose(file);
        return 10;
    }

    if (A->H == 0 || A->W == 0) 
    {
        fclose(file); 
        return 10;
    }

    B->H = A->H; C->H = A->H;
    B->W = A->W; C->W = A->W;

    size_t matrix_size = (A->H) * (A->W);
    A->start = malloc(matrix_size);
    if (A->start == NULL)
    {
        fclose(file);
        return 10;
    }

    B->start = malloc(matrix_size);
    if (B->start == NULL)
    {
        free(A->start);
        fclose(file);
        return 10;
    }
    
    C->start = malloc(matrix_size);
    if (C->start == NULL)
    {
        free(A->start);
        free(B->start);
        fclose(file);
        return 10;
    }

    for (size_t i = 0; i < matrix_size; ++i) 
    {
        if (fread(&(A->start[i]), sizeof(uint8_t), 1, file) != 1
        ||  fread(&(B->start[i]), sizeof(uint8_t), 1, file) != 1
        ||  fread(&(C->start[i]), sizeof(uint8_t), 1, file) != 1)
        {
            free(A->start);
            free(B->start);
            free(C->start);
            fclose(file);
            return 10;
        }
    }

    if (fread(&(D->DH), sizeof(uint16_t), 1, file) != 1
    ||  fread(&(D->DW), sizeof(uint16_t), 1, file) != 1)
    {
        free(A->start);
        free(B->start);
        free(C->start);
        fclose(file);
        return 10;
    }

    if (D->DH == 0 || D->DW == 0) 
    {
        free(A->start);
        free(B->start);
        free(C->start);
        fclose(file);
        return 10;
    }

    size_t matrix_kernel_size = (D->DH) * (D->DW);
    D->start = malloc(matrix_kernel_size);

    if (D->start == NULL)
    {
        free(A->start);
        free(B->start);
        free(C->start);
        fclose(file);
        return 10;
    }

    if (fread(D->start, sizeof(int8_t), matrix_kernel_size, file) != matrix_kernel_size) 
    {
        free(A->start);
        free(B->start);
        free(C->start);
        free(D->start);
        fclose(file);
        return 10;
    }

    fclose(file);
    return 0;
}

// Свертка src с ядром kernel, результат в res. src и kernel не изменяются.
// Память под res->start выделяется здесь же и гарантированно освобождается в случае ошибки.
// Возвращает 0 при успехе, 10 при ошибке выделения памяти.
int convolution(const matrix* src, const matrix_kernel* kernel, matrix* res)
{
    res->H = src->H;
    res->W = src->W;

    size_t res_size = (res->H) * (res->W);
    res->start = malloc(res_size);
    if (res->start == NULL) 
    {
        return 10;
    }

    size_t pos = 0;
    
    int h_bias = (kernel->DH) / 2;
    int w_bias = (kernel->DW) / 2;

    // Проход по исходной матрице
    for (int i = 0; i < (int)src->H; ++i) 
    {
        for (int j = 0; j < (int)src->W; ++j) 
        {
            int32_t value = 0;
            size_t pos_d = 0;

            // Проход по ядру: центр ядра в текущей позиции (i, j)
            for (int h_it = i - h_bias; h_it <= i + h_bias; ++h_it) 
            {
                for (int w_it = j - w_bias; w_it <= j + w_bias; ++w_it) 
                {
                    if (!(h_it < 0 || h_it >= (int)src->H || w_it < 0 || w_it >= (int)src->W))
                    {
                        size_t src_idx = (size_t)h_it * src->W + (size_t)w_it;
                        value += (int32_t)src->start[src_idx] * (int32_t)kernel->start[pos_d];
                    }
                    ++pos_d;
                }
            }

            if (value > 255)
            {
                value %= 251;
            }
            else if (value < 0)
            {
                value = -value;
                value %= 241;
            }

            res->start[pos] = (uint8_t)value;

            ++pos;
        }   
    }   

    return 0;
}

// Записывает H, W и результаты Ar, Br, Cr в файл
// Возвращает 0 при успехе, 10 при ошибке.
int write_data(const char* fname, const matrix* Ar, const matrix* Br, const matrix* Cr) 
{
    FILE* file = fopen(fname, "wb");
    if (file == NULL) 
    {
        return 10;
    }

    if (fwrite(&(Ar->H), sizeof(uint32_t), 1, file) != 1
    ||  fwrite(&(Ar->W), sizeof(uint32_t), 1, file) != 1)
    {
        fclose(file);
        return 10;
    }

    size_t matrix_size = (Ar->H) * (Ar->W);
    for (size_t i = 0; i < matrix_size; ++i) 
    {
        if (fwrite(&(Ar->start[i]), sizeof(uint8_t), 1, file) != 1
        ||  fwrite(&(Br->start[i]), sizeof(uint8_t), 1, file) != 1
        ||  fwrite(&(Cr->start[i]), sizeof(uint8_t), 1, file) != 1)
        {
            fclose(file);
            return 10;
        }
    }

    fclose(file);
    return 0;
}

//Пример запуска программы: ./prog -o output.dat -i ../test/input.bin
int main (int argc, char** argv) 
{
    if (argc != 5)
    {
        return 10;
    }

    const char* input_file  = NULL;
    const char* output_file = NULL;

    for (int i = 1; i < argc - 1; i += 2)
    {
        if (strcmp(argv[i], "-i") == 0) {
            input_file = argv[i + 1];
        } else if (strcmp(argv[i], "-o") == 0) {
            output_file = argv[i + 1];
        } else {
            return 10;
        }
    }

    if (input_file == NULL || output_file == NULL) {
        return 10;
    }

    matrix A = {0}, B = {0}, C = {0};
    matrix_kernel D = {0};
    matrix Ar = {0}, Br = {0}, Cr = {0};

    int err = read_data(input_file, &A, &B, &C, &D);

    if (err != 0) 
    {
        return err;
    }

    err = convolution(&A, &D, &Ar);
    if (err != 0) goto cleanup;

    err = convolution(&B, &D, &Br);
    if (err != 0) goto cleanup;

    err = convolution(&C, &D, &Cr);
    if (err != 0) goto cleanup;

    err = write_data(output_file, &Ar, &Br, &Cr);

cleanup:
    free(A.start); free(B.start); free(C.start);
    free(D.start);
    free(Ar.start); free(Br.start); free(Cr.start);
    return err;
}