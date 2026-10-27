#pragma once

// The BatchPut kernel includes this CANN header but invokes no device atomic
// functions. The scalar CPU control-flow model needs no atomic implementation.
