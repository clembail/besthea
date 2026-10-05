#ifndef INCLUDE_BESTHEA_TIMER_METAL_H_
#define INCLUDE_BESTHEA_TIMER_METAL_H_

#include "besthea/settings.h"

#include <chrono>
#include <iostream>

namespace besthea {
  namespace tools {
    class timer_metal;
  }
}

/*!
 * Class measuring elapsed time between events on a Metal device.
 */
class besthea::tools::timer_metal {
 public:
  /*!
   * Default constructor.
   */
  timer_metal( ) {
    this->was_inited = false;
    this->init( nullptr );
  }

  /*!
   * Constructor with command queue pointer.
   * @param[in] queue_ Metal command queue or context pointer.
   */
  timer_metal( void * queue_ ) {
    this->was_inited = false;
    this->init( queue_ );
  }

  /*!
   * Destructor.
   */
  ~timer_metal( ) {
    destroy( );
  }

  /*!
   * Initialization method.
   * @param[in] queue_ Metal command queue or context pointer.
   */
  void init( void * queue_ ) {
    this->queue = queue_;
    reset( );
    this->was_inited = true;
  }

  /*!
   * Destroys timer resources.
   */
  void destroy( ) {
    this->was_inited = false;
  }

  /*!
   * Submits a start event/timestamp marking the start of the measured timespan.
   */
  void start_submit( ) {
    this->start_tp = std::chrono::steady_clock::now( );
  }

  /*!
   * Submits a stop event/timestamp marking the end of the measured timespan.
   */
  void stop_submit( ) {
    this->stop_tp = std::chrono::steady_clock::now( );
    this->was_time_collected = false;
  }

  /*!
   * Resets the timer.
   */
  void reset( ) {
    this->elapsed_time = 0.0;
    this->was_time_collected = true;
  }

  /*!
   * Synchronizes and returns elapsed time in seconds.
   */
  double get_elapsed_time_in_seconds( ) {
    if ( !was_time_collected ) {
      collect_time( );
      this->was_time_collected = true;
    }
    return this->elapsed_time;
  }

 private:
  /*!
   * Collects elapsed time.
   */
  void collect_time( ) {
    std::chrono::duration< double > diff = this->stop_tp - this->start_tp;
    this->elapsed_time += diff.count( );
  }

 private:
  void * queue;                                            //!< Used Metal queue
  std::chrono::steady_clock::time_point start_tp;          //!< Start time point
  std::chrono::steady_clock::time_point stop_tp;           //!< Stop time point
  double elapsed_time;                                     //!< Elapsed time in seconds
  bool was_time_collected;
  bool was_inited;
};

#endif /* INCLUDE_BESTHEA_TIMER_METAL_H_ */
