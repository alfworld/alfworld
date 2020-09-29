;; Specification in PDDL1 of the Question domain

(define (domain put_task)
 (:requirements
    :adl
    :action-costs
    :typing
 )
 (:types
  agent
  location
  receptacle
  object
  rtype
  otype
  )


 (:predicates
    (atLocation ?a - agent ?l - location)                     ; true if the agent is at the location
    (receptacleAtLocation ?r - receptacle ?l - location)      ; true if the receptacle is at the location (constant)
    (objectAtLocation ?o - object ?l - location)              ; true if the object is at the location
    (openable ?r - receptacle)                                ; true if a receptacle is openable
    (opened ?r - receptacle)                                  ; true if a receptacle is opened
    (inReceptacle ?o - object ?r - receptacle)                ; object ?o is in receptacle ?r
    (checked ?r - receptacle)                                 ; whether the receptacle has been looked inside/visited
    (examined ?l - location)                                 ; TODO
    (receptacleType ?r - receptacle ?t - rtype)               ; the type of receptacle (Cabinet vs Cabinet|01|2...)
    (objectType ?o - object ?t - otype)                       ; the type of object (Apple vs Apple|01|2...)
    (holds ?a - agent ?o - object)                            ; object ?o is held by agent ?a
    (holdsAny ?a - agent)                                     ; agent ?a holds an object
    (full ?r - receptacle)                                    ; true if the receptacle has no remaining space
 )

  (:functions
    (distance ?from ?to)
    (total-cost) - number
   )

;; All actions are specified such that the final arguments are the ones used
;; for performing actions in Unity.


(:action look
    :parameters (?a - agent ?l - location)
    :precondition
        (and
            (atLocation ?a ?l)
        )
    :effect
        (and
            (checked ?l)
        )
)

(:action inventory
    :parameters (?a - agent)
    :precondition
        ()
    :effect
        (and
            (checked ?a)
        )
)

(:action examineReceptacle
    :parameters (?a - agent ?r - receptacle)
    :precondition
        (and
            (exists (?l - location)
                (and
                    (atLocation ?a ?l)
                    (receptacleAtLocation ?r ?l)
                )
            )
        )
    :effect
        (and
            (checked ?r)
        )
)

; (:action examineObject
;     :parameters (?a - agent ?o - object)
;     :precondition
;         (and
;             (exists (?l - location)
;                 (and
;                     (atLocation ?a ?l)
;                     (objectAtLocation ?o ?l)
;                 )
;             )
;         )
;     :effect
;         (and
;             (checked ?o)
;         )
; )

;; agent goes to receptacle
 (:action GotoLocation
    :parameters (?a - agent ?lStart - location ?lEnd - location)
    :precondition (atLocation ?a ?lStart)
    :effect (and
                (not (atLocation ?a ?lStart))
                (atLocation ?a ?lEnd)
                ; (forall (?r - receptacle)
                ;     (when (and (receptacleAtLocation ?r ?lEnd)
                ;                (or (not (openable ?r)) (opened ?r)))
                ;         (checked ?r)
                ;     )
                ; )
                ; (increase (total-cost) (distance ?lStart ?lEnd))
                (increase (total-cost) 1)
            )
 )

;; agent opens receptacle
 (:action OpenObject
    :parameters (?a - agent ?l - location ?r - receptacle)
    :precondition (and
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (openable ?r)
            (forall (?re - receptacle)
                (not (opened ?re)))
            )
    :effect (and
                (opened ?r)
                (checked ?r)
                (increase (total-cost) 1)
            )
 )
;; agent closes receptacle
 (:action CloseObject
    :parameters (?a - agent ?l - location ?r - receptacle)
    :precondition (and
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (openable ?r)
            (opened ?r)
            )
    :effect (and
                (not (opened ?r))
                (increase (total-cost) 1)
            )

 )

;; agent picks up object
 (:action PickupObject
    :parameters (?a - agent ?l - location ?o - object ?r - receptacle)
    :precondition (and
            (atLocation ?a ?l)
            ;(receptacleAtLocation ?r ?l)
            (objectAtLocation ?o ?l)
            (or (not (openable ?r)) (opened ?r))    ; receptacle is opened if it is openable
            (inReceptacle ?o ?r)
            (not (holdsAny ?a))
            )
    :effect (and
                (not (inReceptacle ?o ?r))
                (holds ?a ?o)
                (holdsAny ?a)
                (not (objectAtLocation ?o ?l))
                ;(not (full ?r))
                (increase (total-cost) 1)
            )
 )

; agent picks up object
 (:action PickupObjectNoReceptacle
    :parameters (?a - agent ?l - location ?o - object)
    :precondition (and
            (atLocation ?a ?l)
            (objectAtLocation ?o ?l)
            (forall (?r - receptacle)
                ;(or
                    (not (inReceptacle ?o ?r))
                    ;(or (not (openable ?r)) (opened ?r))    ; receptacle is opened if it is openable
                ;)
            )
            (not (holdsAny ?a))
            )
    :effect (and
                (holds ?a ?o)
                (holdsAny ?a)
                (not (objectAtLocation ?o ?l))

                (increase (total-cost) 1)
            )
 )

;; agent puts down an object
 (:action PutObject
    :parameters (?a - agent ?l - location ?ot - otype ?o - object ?r - receptacle)
    :precondition (and
            (atLocation ?a ?l)
            (receptacleAtLocation ?r ?l)
            (or (not (openable ?r)) (opened ?r))    ; receptacle is opened if it is openable
            (not (full ?r))
            (objectType ?o ?ot)
            (holds ?a ?o)
            )
    :effect (and
                (inReceptacle ?o ?r)
                (full ?r)
                (not (holds ?a ?o))
                (not (holdsAny ?a))
                (increase (total-cost) 1)
            )
 )

)


