/**
  ******************************************************************************
  * @file    layers.h
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

#ifndef LAYERS_H
#define LAYERS_H
#pragma once
 
#include "layers_common.h"
#include "layers_conv2d.h"
#include "layers_custom.h"
#include "layers_dense.h"
#include "layers_dense_dqnn.h"
#include "layers_formats_converters.h"
#include "layers_generic.h"
#include "layers_lite_graph.h"
#include "layers_nl.h"
#include "layers_norm.h"
#include "layers_pool.h"
#include "layers_rnn.h"
#include "layers_sm.h"
#include "layers_ml.h"
#include "layers_ml_iforest.h"
#include "layers_ml_svc.h"
#include "layers_ml.h"
#include "layers_ml_linearclassifier.h"
#include "layers_ml_treeensembleclassifier.h"
#include "layers_ml_treeensembleregressor.h"
#include "layers_ml_svmregressor.h"

#include "layers_conv2d_dqnn.h"
#include "layers_pool_dqnn.h"
#include "layers_generic_dqnn.h"

// #include "layers_template.h"

#ifdef USE_OPERATORS
  #include "layers_lambda.h"
#endif /* USE_OPERATORS */


AI_API_DECLARE_BEGIN

/*!
 * @struct ai_any_layer_ptr
 * @ingroup layers
 * @brief Generic union for typed layers pointers
 */
typedef struct {
  ai_layer_type type;              /*!< layer type id (see @ref ai_layer_type) */
  union {
#define LAYER_ENTRY(type_, id_, struct_, forward_func_, init_func_, destroy_func_) \
   AI_CONCAT(ai_layer_, struct_)* struct_;
#include "layers_list.h"
  };
} ai_any_layer_ptr;


AI_API_DECLARE_END

#endif /*LAYERS_H*/
