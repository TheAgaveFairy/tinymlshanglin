/**
  ******************************************************************************
  * @file    layers_common.h
  * @author  AST Embedded Analytics Research Platform
  * @brief   header file of AI platform layers datatypes
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2017 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  @verbatim
  @endverbatim
  ******************************************************************************
  */

#ifndef LAYERS_COMMON_H
#define LAYERS_COMMON_H
#pragma once

// #include <stdlib.h>

#ifdef USE_CYCLE_MEASUREMENTS
  #include "layers_cycles_estimation.h"
#endif
#include "ai_platform.h"
#include "ai_common_config.h"

#include "core_common.h"

/* optimizations */
#define AI_OPTIM_DICT8_DOT_ARRAY_F32    (1)
#define AI_OPTIM_DICT8_DTCM             (1)
#define AI_OPTIM_FUNC_MP_ARRAY_F32      (0)


#define AI_LAYER_OBJ(obj_) \
  ((ai_layer_base*)(obj_))

#define AI_LAYER_FUNC(func_) \
  ((layer_func)(func_))

#define AI_LAYER_TYPE(type_) \
  ( (ai_layer_type)((ai_u32)(type_)&0xFFFF) )

#define AI_LAYER_TYPE_ENTRY(type_) \
   AI_CONCAT(AI_CONCAT(AI_LAYER_, type_), _TYPE)

#define AI_LAYER_TYPE_NAME(type_) \
  ai_layer_type_name(AI_LAYER_TYPE(type_))


#if (AI_TOOLS_API_VERSION <= AI_TOOLS_API_VERSION_1_3)
#pragma message ("Including deprecated AI_LAYER_OBJ_INIT, AI_LAYER_OBJ_DECLARE")

AI_DEPRECATED
#define AI_LAYER_OBJ_INIT(type_, id_, network_, \
                          next_, forward_, ...) \
  { \
    AI_NODE_COMMON_INIT(AI_CONCAT(AI_LAYER_, type_), id_, 0x0, \
                        NULL, network_, next_, forward_), \
    ## __VA_ARGS__ \
  }

AI_DEPRECATED
#define AI_LAYER_OBJ_DECLARE(varname_, id_, type_, struct_, forward_func_, \
                             network_, next_, attr_, ...) \
    AI_ALIGNED(4) \
    attr_ AI_CONCAT(ai_layer_, struct_) varname_ = \
    AI_LAYER_OBJ_INIT(type_, id_, network_, \
                      next_, forward_func_, \
                      ## __VA_ARGS__);

#else

#define AI_LAYER_OBJ_INIT(type_, id_, flags_, klass_, network_, \
                          next_, forward_, tensors_, ...) \
  { \
    AI_NODE_COMMON_INIT(AI_CONCAT(AI_LAYER_, type_), id_, flags_, \
                        klass_, network_, next_, forward_), \
    .tensors = (tensors_), \
    ## __VA_ARGS__ \
  }

#define AI_LAYER_OBJ_DECLARE( \
  varname_, id_, \
  type_, flags_, klass_obj_, \
  struct_, forward_func_, \
  tensors_chain_, \
  network_, next_, attr_, ...) \
  AI_ALIGNED(4) \
  attr_ AI_CONCAT(ai_layer_, struct_) varname_ = \
    AI_LAYER_OBJ_INIT(type_, id_, flags_, klass_obj_, network_, \
                      next_, forward_func_, tensors_chain_, ## __VA_ARGS__);

#endif      /* AI_TOOLS_API_VERSION_1_3 */

#ifdef HAS_AI_ASSERT
  #define AI_LAYER_IO_GET(layer_, in_, out_) \
    ASSERT_LAYER_SANITY(layer_) \
    const ai_tensor* in_  = GET_TENSOR_IN((layer_)->tensors, 0); \
    ai_tensor* out_ = GET_TENSOR_OUT((layer_)->tensors, 0); \
    ASSERT_TENSOR_DATA_SANITY(in_) \
    ASSERT_TENSOR_DATA_SANITY(out_)

  #define AI_LAYER_TENSOR_LIST_IO_GET(layer_, tlist_in_, tlist_out_) \
    ASSERT_LAYER_SANITY(layer_) \
    const ai_tensor_list* tlist_in_  = GET_TENSOR_LIST_IN((layer_)->tensors); \
    ai_tensor_list* tlist_out_ = GET_TENSOR_LIST_OUT((layer_)->tensors); \
    ASSERT_TENSOR_LIST_SANITY(tlist_in_) \
    ASSERT_TENSOR_LIST_SANITY(tlist_out_)

  #define AI_LAYER_WEIGHTS_GET(layer_, weights_, bias_) \
    const ai_tensor* weights_  = GET_TENSOR_WEIGHTS((layer_)->tensors, 0); \
    const ai_tensor* bias_ = (GET_TENSOR_LIST_SIZE(GET_TENSOR_LIST_WEIGTHS((layer_)->tensors))>1) \
                                ? GET_TENSOR_WEIGHTS((layer_)->tensors, 1) \
                                : NULL; \
    ASSERT_TENSOR_DATA_SANITY(weights_) \
    if (bias_) { ASSERT_TENSOR_DATA_SANITY(bias_) }
#else
  #define AI_LAYER_IO_GET(layer_, in_, out_) \
    const ai_tensor* in_  = GET_TENSOR_IN((layer_)->tensors, 0); \
    ai_tensor* out_ = GET_TENSOR_OUT((layer_)->tensors, 0);

  #define AI_LAYER_TENSOR_LIST_IO_GET(layer_, tlist_in_, tlist_out_) \
    const ai_tensor_list* tlist_in_  = GET_TENSOR_LIST_IN((layer_)->tensors); \
    ai_tensor_list* tlist_out_ = GET_TENSOR_LIST_OUT((layer_)->tensors);

  #define AI_LAYER_WEIGHTS_GET(layer_, weights_, bias_) \
    const ai_tensor* weights_  = GET_TENSOR_WEIGHTS((layer_)->tensors, 0); \
    const ai_tensor* bias_ = (GET_TENSOR_LIST_SIZE(GET_TENSOR_LIST_WEIGTHS((layer_)->tensors))>1) \
                                ? GET_TENSOR_WEIGHTS((layer_)->tensors, 1) \
                                : NULL; \

#endif  /*HAS_AI_ASSERT*/
    

AI_API_DECLARE_BEGIN

/*!
 * @defgroup layers_common Layers Common
 * @brief Implementation of the common layers datastructures 
 * This header enumerates the layers specific definition implemented in the
 * library toghether with the macros and datatypes used to manipulate them. 
 */

/*!
 * @typedef (*func_copy_tensor)
 * @ingroup layers_common
 * @brief Fuction pointer for generic tensor copy routines
 * this function pointer abstracts a generic tensor copy routine.
 */
typedef ai_bool (*func_copy_tensor)(ai_tensor* dst, const ai_tensor* src);

/*!
 * @enum ai_layer_type
 * @ingroup layers_common
 * @brief ai_tools supported layers type id
 */
typedef enum {
#define LAYER_ENTRY(type_, id_, struct_, forward_func_, init_func_, destroy_func_) \
   AI_LAYER_TYPE_ENTRY(type_) = id_,
#include "layers_list.h"
} ai_layer_type;

#define AI_LAYER_COMMON_FIELDS_DECLARE \
  AI_NODE_COMMON_FIELDS_DECLARE

#define AI_LAYER_STATEFUL_FIELDS_DECLARE \
  AI_NODE_STATEFUL_FIELDS_DECLARE


/*!
 * @typedef void (*layer_func)(struct ai_layer_* layer)
 * @ingroup layers_common
 * @brief Callback signatures for all layers forward functions
 */
typedef node_func          layer_func;

/*!
 * @struct ai_layer_base
 * @ingroup layers_common
 * @brief Structure encoding a base layer in the network
 *
 */
typedef ai_node             ai_layer_base;

/*!
 * @struct ai_layer_stateful
 * @ingroup layers_common
 * @brief Structure encoding a stateful layer in the network
 *
 */
typedef ai_node_stateful    ai_layer_stateful;

/*!
 * @brief Check the custom network types against the internally compiled ones
 * Helper function to check if the private APIs where compiled with a different
 * `datatypes_network.h` than the one provided to the caller.
 * @ingroup layers_common
 * @param signatures list of type sizes signatures (first element is the number of types)
 * @return false if there is a type size mismatch
 */
AI_INTERNAL_API
ai_bool ai_check_custom_types(const ai_custom_type_signature* signatures);

/*!
 * @brief Helper API to retrieve a human readable layer type from enum
 * @ingroup layers_common
 * @param type in type of layer
 * @return string defining the type of the layer
 */
AI_INTERNAL_API
const char* ai_layer_type_name(const ai_layer_type type);

/*!
 * @brief Helper API to check if a node is a valid layer type
 * @ingroup layers_common
 * @param type in type of layer
 * @return true if the layer is one of the ones listed in the enum, 
 * false otherwise
 */
AI_INTERNAL_API
ai_bool ai_layer_type_is_valid(const ai_layer_type type);

AI_API_DECLARE_END

#endif /*LAYERS_COMMON_H*/
