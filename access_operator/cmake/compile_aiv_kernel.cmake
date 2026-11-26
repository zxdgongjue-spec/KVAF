cmake_minimum_required(VERSION 3.18)

# Invoked as cmake -P <script> -- <compiler> <arguments>.
math(EXPR last_arg "${CMAKE_ARGC} - 1")
foreach(index RANGE 4 ${last_arg})
  list(APPEND command "${CMAKE_ARGV${index}}")
endforeach()

execute_process(COMMAND ${command}
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE output)
if(NOT result STREQUAL "0")
  message(FATAL_ERROR "AIV compilation failed (${result})\n${output}")
endif()

# Bisheng can emit this informational line repeatedly for one source location.
# Keep the compiler's NOP insertion pass and all other diagnostics.
if(NOT "$ENV{KVAF_AIV_COMPILER_VERBOSE}" STREQUAL "1")
  set(nop_note "note: [^\r\n]+:[0-9]+:[0-9]+: Identify the scenario where the NOP instruction is inserted to prevent a running failure[.]")
  string(REGEX REPLACE "(^|\n)(${nop_note}(\r?\n|$))+" "\\1" output "${output}")
endif()
if(NOT output STREQUAL "")
  message("${output}")
endif()
